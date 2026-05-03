#!/usr/bin/env python3
"""serial_to_wav.py — capture an audio_dump payload from the device's UART
and convert it back to a 16 kHz mono WAV file.

Usage:
    # Read live from a serial port:
    python3 serial_to_wav.py --port /dev/cu.usbmodem101 --out dump.wav

    # Or pipe a saved log through stdin:
    idf.py monitor | tee monitor.log
    python3 serial_to_wav.py --in monitor.log --out dump.wav

The script looks for the framing emitted by audio_dump.c:

    ===AUDIO_DUMP_BEGIN n_samples=NNN rate=RRRR===
    <base64 lines>
    ===AUDIO_DUMP_END===

When it sees BEGIN, it accumulates base64 lines until END, decodes them
into a flat byte stream of int16 little-endian PCM, and writes a WAV.

If multiple dumps appear in the input, only the last one is written
(unless --all is passed, in which case dump_001.wav, dump_002.wav, ...
are produced).

Requires only the standard library — no pyserial dep when reading from
stdin or a file. pyserial is needed for --port; the script will tell you
to `pip install pyserial` if missing.
"""

from __future__ import annotations

import argparse
import base64
import re
import struct
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


BEGIN_RE = re.compile(r"===AUDIO_DUMP_BEGIN\s+n_samples=(\d+)\s+rate=(\d+)===")
END_MARK = "===AUDIO_DUMP_END==="


@dataclass
class Dump:
    n_samples: int
    rate: int
    pcm_bytes: bytes


def line_iter_from_serial(port: str, baud: int) -> Iterator[str]:
    try:
        import serial  # type: ignore
    except ImportError:
        sys.exit("error: --port requires pyserial; run "
                 "`pip install pyserial`")
    s = serial.Serial(port, baud, timeout=1)
    print(f"reading from {port} @ {baud}; press Ctrl-C to stop after a dump",
          file=sys.stderr)
    try:
        buf = bytearray()
        while True:
            data = s.read(4096)
            if not data:
                continue
            buf.extend(data)
            while b"\n" in buf:
                line, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                yield line.decode("utf-8", errors="replace").rstrip("\r")
    finally:
        s.close()


def line_iter_from_file(path: str) -> Iterator[str]:
    if path == "-":
        for line in sys.stdin:
            yield line.rstrip("\r\n")
    else:
        with open(path, "r", errors="replace") as f:
            for line in f:
                yield line.rstrip("\r\n")


def extract_dumps(lines: Iterable[str]) -> Iterator[Dump]:
    state = "idle"
    n_samples = 0
    rate = 0
    b64_chunks: list[str] = []
    for line in lines:
        if state == "idle":
            m = BEGIN_RE.search(line)
            if m:
                n_samples = int(m.group(1))
                rate      = int(m.group(2))
                b64_chunks = []
                state = "in_dump"
        elif state == "in_dump":
            if END_MARK in line:
                joined = "".join(b64_chunks)
                try:
                    pcm = base64.b64decode(joined)
                except Exception as e:
                    print(f"warning: base64 decode failed: {e}; skipping",
                          file=sys.stderr)
                    state = "idle"
                    continue
                expected = n_samples * 2
                if len(pcm) != expected:
                    print(f"warning: decoded {len(pcm)} bytes, "
                          f"expected {expected} (n_samples={n_samples})",
                          file=sys.stderr)
                yield Dump(n_samples, rate, pcm)
                state = "idle"
            else:
                # Strip whitespace from base64 lines; ignore non-b64 chars
                # (e.g. ESP_LOG output that snuck in).
                cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", line)
                if cleaned:
                    b64_chunks.append(cleaned)


def write_wav(path: str, dump: Dump) -> None:
    n_bytes = len(dump.pcm_bytes)
    header = b"RIFF"
    header += struct.pack("<I", 36 + n_bytes)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH",
                          16,                  # fmt chunk size
                          1,                   # PCM
                          1,                   # mono
                          dump.rate,           # sample rate
                          dump.rate * 2,       # byte rate
                          2,                   # block align
                          16)                  # bits/sample
    header += b"data"
    header += struct.pack("<I", n_bytes)
    with open(path, "wb") as f:
        f.write(header)
        f.write(dump.pcm_bytes)
    print(f"wrote {path}: {dump.n_samples} samples, "
          f"{dump.n_samples / dump.rate:.3f} s @ {dump.rate} Hz")


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="serial device, e.g. /dev/cu.usbmodem101")
    src.add_argument("--in", dest="infile",
                     help="read framed output from a file (or '-' for stdin)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--out", default="dump.wav",
                   help="output WAV (or template if --all; default dump.wav)")
    p.add_argument("--all", action="store_true",
                   help="write every dump found, not just the last")
    args = p.parse_args()

    if args.port:
        lines = line_iter_from_serial(args.port, args.baud)
    else:
        lines = line_iter_from_file(args.infile)

    last: Optional[Dump] = None
    count = 0
    for dump in extract_dumps(lines):
        count += 1
        if args.all:
            base = args.out
            if base.lower().endswith(".wav"):
                path = base[:-4] + f"_{count:03d}.wav"
            else:
                path = base + f"_{count:03d}.wav"
            write_wav(path, dump)
        else:
            last = dump

    if not args.all:
        if last is None:
            sys.exit("no AUDIO_DUMP frame found in input")
        write_wav(args.out, last)


if __name__ == "__main__":
    main()
