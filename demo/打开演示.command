#!/bin/zsh
set -e
demo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$demo_dir"
/usr/bin/python3 server.py
