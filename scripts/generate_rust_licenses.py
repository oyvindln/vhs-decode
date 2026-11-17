#!/usr/bin/env python3
"""Write the licence texts of every Rust crate linked into vhsd_rust.

Usage: python scripts/generate_rust_licenses.py [output file]

Collects the LICENSE/COPYING/NOTICE files that each crate ships, for the
crates reachable through normal dependencies on any of the release targets.
"""

import json
import os
import subprocess
import sys

TARGETS = [
    "x86_64-pc-windows-msvc",
    "aarch64-pc-windows-msvc",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
]
LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "copyright", "unlicense")

MIT_TEXT = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def crates_for_target(target):
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--filter-platform", target]
        )
    )
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, stack = set(), [root]
    while stack:
        pkg = stack.pop()
        if pkg in seen:
            continue
        seen.add(pkg)
        for dep in nodes[pkg]["deps"]:
            if any(k["kind"] in (None, "normal") for k in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.discard(root)
    return [packages[i] for i in seen]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "THIRD_PARTY_LICENSES_RUST.txt"
    crates = {}
    for target in TARGETS:
        for p in crates_for_target(target):
            crates[(p["name"], p["version"])] = p

    apache_text = None
    for p in crates.values():
        crate_dir = os.path.dirname(p["manifest_path"])
        for f in ("LICENSE-APACHE", "LICENSE.APACHE", "LICENSE-APACHE-2.0"):
            path = os.path.join(crate_dir, f)
            if apache_text is None and os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as src:
                    apache_text = src.read().strip()

    missing = []
    needs_apache = False
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("Licences of the Rust crates compiled into vhsd_rust.\n\n")
        for (name, version), p in sorted(crates.items()):
            out.write("=" * 78 + "\n")
            out.write(f"{name} {version}\nLicense: {p.get('license')}\n")
            if p.get("repository"):
                out.write(f"Source: {p['repository']}\n")
            crate_dir = os.path.dirname(p["manifest_path"])
            files = sorted(
                f for f in os.listdir(crate_dir)
                if f.lower().startswith(LICENSE_PREFIXES) and os.path.isfile(os.path.join(crate_dir, f))
            )
            if not files:
                # The crate package ships no licence file; use the standard
                # text of the licence it declares.
                lic = p.get("license") or ""
                holders = ", ".join(a.split(" <")[0] for a in p.get("authors", []))
                holders = holders or f"the {p.get('repository') or name} contributors"
                if "MIT" in lic:
                    out.write("\n--- MIT License (standard text) ---\n\n")
                    out.write(f"Copyright (c) {holders}\n\n{MIT_TEXT}\n")
                elif "Apache-2.0" in lic and apache_text:
                    out.write(f"\nCopyright (c) {holders}\n")
                    out.write("Licensed under the Apache License, Version 2.0; see the end of this file.\n")
                    needs_apache = True
                else:
                    missing.append(f"{name} {version}")
            for f in files:
                with open(os.path.join(crate_dir, f), encoding="utf-8", errors="replace") as src:
                    out.write(f"\n--- {f} ---\n\n{src.read().strip()}\n")
            out.write("\n")
        if needs_apache:
            out.write("=" * 78 + "\nApache License, Version 2.0\n\n" + apache_text + "\n")
    print(f"Wrote {len(crates)} crates to {out_path}")
    if missing:
        print("No licence file shipped by:", ", ".join(missing))


if __name__ == "__main__":
    main()
