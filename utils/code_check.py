#!/usr/bin/env python3
"""
Run linter code check on all or selected files.

Example usage:
    python tools/code_check.py            # checks repository root (.)
    python tools/code_check.py -p path    # checks the file or directory at path
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_EXCLUDES = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".eggs"}


def find_py_files(dirpath: Path):
    files = []
    for p in dirpath.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDES for part in p.parts):
            continue
        files.append(p)
    return files


def run_linter(cmd_args):
    try:
        result = subprocess.run(cmd_args, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"Command not found: {cmd_args[0]}", file=sys.stderr)
        return 127


def main():
    parser = argparse.ArgumentParser(
        description="Run flake8/pflake8 on the tree or a specific path."
    )
    parser.add_argument(
        "-p",
        "--path",
        default=".",
        help="Path (file or directory) to lint. Default: repository root.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    target = Path(args.path).resolve()
    if not target.exists():
        print(f"Path does not exist: {target}", file=sys.stderr)
        sys.exit(2)

    # gather targets
    if target.is_file() and target.suffix == ".py":
        targets = [target]
    elif target.is_file():
        # flake8 will ignore non-py
        targets = [target]
    else:
        targets = find_py_files(target)

    # prepare short relative paths for printing and passing to linter
    rel_targets = []
    for p in targets:
        try:
            rel = p.relative_to(cwd)
        except Exception:
            rel = p
        rel_targets.append(str(rel))

    cmd = ["pflake8"] + rel_targets
    rc = run_linter(cmd)
    if rc:
        msg = "\n❌ FAILURE: Issues detected."
    else:
        msg = "\n✅ SUCCESS: No linting or formatting issues found."
    print(msg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
