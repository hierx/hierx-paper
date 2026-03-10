#!/usr/bin/env python3
"""Local HTTP server with Range request and CORS support for PMTiles.

Python's built-in http.server doesn't support Range requests, which are
required for PMTiles to work (it fetches byte ranges from the tile archive).

Usage:
    python interactive_maps/serve.py              # serves interactive_maps/ on :8080
    python interactive_maps/serve.py --port 3000  # custom port
    python interactive_maps/serve.py --dir .      # custom directory
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler supporting Range requests and CORS headers."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET with optional Range header."""
        range_header = self.headers.get("Range")
        if not range_header:
            super().do_GET()
            return

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        file_size = os.path.getsize(path)

        # Parse Range: bytes=start-end
        try:
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
        except (ValueError, IndexError):
            self.send_error(416, "Invalid range")
            return

        if start >= file_size or end >= file_size or start > end:
            self.send_error(416, "Range not satisfiable")
            return

        content_length = end - start + 1

        self.send_response(206)
        content_type = self.guess_type(path)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(content_length))

    def log_message(self, format: str, *args: object) -> None:
        """Quieter logging: skip 200/206 for tile requests."""
        if len(args) >= 2 and str(args[1]) in ("200", "206") and ".pmtiles" in str(args[0]):
            return
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("interactive_maps"),
        help="Directory to serve (default: interactive_maps/)",
    )
    args = parser.parse_args()

    directory = str(args.dir.resolve())
    handler = partial(RangeRequestHandler, directory=directory)
    server = HTTPServer(("0.0.0.0", args.port), handler)

    print(f"Serving {directory} at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
