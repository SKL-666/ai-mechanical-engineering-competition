#!/usr/bin/env python3
"""Serve the offline demo on localhost with HTTP byte-range support for MP4 seek."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import webbrowser

DEMO = Path(__file__).resolve().parent


class RangeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO), **kwargs)

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()
        source = path.open("rb")
        size = path.stat().st_size
        requested = self.headers.get("Range")
        start, end = 0, size - 1
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested.strip())
            if not match:
                source.close()
                self.send_error(416, "Invalid range")
                return None
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
            if start >= size or end < start:
                source.close()
                self.send_error(416, "Range outside file")
                return None
            end = min(end, size - 1)
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        source.seek(start)
        self.remaining = end - start + 1
        return source

    def copyfile(self, source, output):
        while self.remaining > 0:
            chunk = source.read(min(self.remaining, 128 * 1024))
            if not chunk:
                break
            output.write(chunk)
            self.remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free localhost port")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), RangeHandler)
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    print(f"演示地址：{url}\n保持此窗口运行；按 Ctrl+C 结束。", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
