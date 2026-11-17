"""Fail if a built wheel carries a path from the machine that built it.

The bug this guards against shipped once already. `crates/grilly-core/
build.rs` compiled the kernels into `OUT_DIR` and handed the crate that
directory as `GRILLY_CORE_SPV_DIR`; `device.rs` read it at bring-up with
`env!`. The built extension therefore contained a literal

    .../target/release/build/grilly-core-d61c8e3d3b141090/out/spv

— cargo hash and all. On the machine that built it, that directory exists
and everything works, which is why no amount of local testing finds this.
On any other machine the device opens, finds no kernels, and fails. A wheel
built in CI would have been dead on arrival for every user.

A development machine cannot see this bug. CI can, in one grep. That is the
whole argument for having CI on a project whose real tests need a GPU no
runner has.

Usage: check_relocatable.py <dist-dir>
"""

from __future__ import annotations

import pathlib
import re
import sys
import zipfile

#: paths that only exist on a build machine. `target/.../build/<crate>-<hash>`
#: is cargo's; the others catch a wheel built in a container or a home dir.
SUSPECT = [
    re.compile(rb"target[/\\](debug|release)[/\\]build[/\\]"),
    re.compile(rb"[/\\]home[/\\]runner[/\\]work[/\\]"),
    re.compile(rb"[A-Za-z]:[/\\]Users[/\\][^/\\]+[/\\]Documents[/\\]"),
    # the GitHub Windows runner's workspace, D:\a\<repo>\<repo>
    re.compile(rb"\b[A-Za-z]:[/\\]a[/\\][^/\\]+[/\\]"),
]

#: maturin's CycloneDX SBOM names each workspace crate by its build-time
#: `path+file://` URL. That is provenance metadata — nothing loads anything
#: from it — so it cannot break an install, and it is skipped.
SKIP = re.compile(r"\.dist-info/sboms/")

# Rust panic messages legitimately embed source paths like
# `crates/grilly-core/src/device.rs`; those are relative and harmless.
TEXT = re.compile(rb"[ -~]{8,}")


def offenders(blob: bytes) -> list:
    hits = []
    for run in TEXT.findall(blob):
        for pattern in SUSPECT:
            if pattern.search(run):
                hits.append(run.decode("ascii", "replace"))
                break
    return hits


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    wheels = sorted(pathlib.Path(argv[1]).glob("*.whl"))
    if not wheels:
        print(f"no wheels in {argv[1]}", file=sys.stderr)
        return 2

    bad = 0
    for wheel in wheels:
        print(f"checking {wheel.name}")
        with zipfile.ZipFile(wheel) as zf:
            for info in zf.infolist():
                if info.is_dir() or SKIP.search(info.filename):
                    continue
                hits = offenders(zf.read(info))
                for hit in hits:
                    print(f"  {info.filename}: build-machine path: {hit}")
                bad += len(hits)

    if bad:
        print(
            f"\n{bad} build-machine path(s) in the wheel. This wheel works only on "
            f"the machine that built it.\nThe fix is to embed the data rather than "
            f"its location — see crates/grilly-core/build.rs, which bakes the "
            f"kernels in with include_bytes!.",
            file=sys.stderr,
        )
        return 1
    print(f"\n{len(wheels)} wheel(s) carry no build-machine paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
