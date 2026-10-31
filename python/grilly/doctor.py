"""`python -m grilly.doctor` — why grilly does or does not work here.

This exists because of CLAUDE.md §8's first trap, which cost this project
hours: *a backend reading of "numpy" does not mean there is no Vulkan tier.*
It usually means the package is not importable from the interpreter being
tested in. That mistake is cheap to make and expensive to debug, and a user
who hits it has no way at all to tell the two apart. So the rule here is to
report **which interpreter, which install, which device**, before reporting
any verdict.

Everything it prints is something a bug report needs and a user cannot
easily get. It never raises: a diagnostic that crashes on a broken install
is diagnosing nothing.

    $ python -m grilly.doctor
    $ grilly-doctor              # same thing, installed as a script
"""

from __future__ import annotations

import os
import platform
import sys

__all__ = ["main", "report"]

OK, BAD, MEH = "ok", "FAIL", "warn"


def _v1_clash() -> "tuple[str, str] | None":
    """Is grilly v1 installed beside grilly2?

    Both distributions own a top-level ``grilly`` package. pip does not
    model that: installing one over the other silently overwrites files,
    and uninstalling either can take the other's with it. Whichever one
    won the race is the one that imports, so the symptom is an
    AttributeError or a missing module rather than anything that says
    "two packages".
    """
    from importlib import metadata

    try:
        v1 = metadata.distribution("grilly")
    except metadata.PackageNotFoundError:
        return None
    try:
        v2 = metadata.distribution("grilly2").version
    except metadata.PackageNotFoundError:
        v2 = None
    return (v1.version, v2)


def _rows() -> list:
    """``(status, label, detail)`` for everything worth knowing."""
    rows = [
        (OK, "python", f"{platform.python_version()} ({sys.executable})"),
        (OK, "platform", f"{platform.system()} {platform.release()} {platform.machine()}"),
    ]

    clash = _v1_clash()
    if clash is not None:
        v1, v2 = clash
        rows.append((
            BAD, "grilly v1 is also installed",
            f"grilly {v1} and grilly2 {v2 or '(not installed)'} both own the `grilly` "
            f"package. Run `pip uninstall grilly` and reinstall grilly2 — whichever "
            f"was installed last is the one importing, and pip will not warn you.",
        ))

    try:
        import grilly
    except Exception as exc:                       # noqa: BLE001 - report anything
        rows.append((BAD, "import grilly", f"{type(exc).__name__}: {exc}"))
        return rows
    rows.append((OK, "import grilly", getattr(grilly, "__file__", "?")))

    try:
        from grilly import _C
    except Exception as exc:                       # noqa: BLE001
        rows.append((
            BAD, "import grilly._C",
            f"{type(exc).__name__}: {exc}\n"
            f"          The compiled extension did not load. On Windows this is "
            f"usually a missing Vulkan loader: `vulkan-1.dll` ships with the GPU "
            f"driver, so install or update the driver for your card.",
        ))
        return rows
    rows.append((OK, "import grilly._C", getattr(_C, "__file__", "?")))

    for var in ("GRILLY_SHADER_DIR", "GRILLY_VERBOSE", "GRILLY_LAUNCH_BLOCKING",
                "VULKAN_SDK", "VK_ICD_FILENAMES", "VK_LOADER_DEBUG"):
        if os.environ.get(var):
            rows.append((MEH, f"${var}", os.environ[var]))

    # not `grilly.cuda.is_available()` — that is False on purpose, so that
    # `"cuda" if torch.cuda.is_available() else "cpu"` picks the path grilly
    # actually serves. It says nothing about whether the GPU works.
    try:
        rows.append((OK, "device", _C.device_name()))
    except Exception as exc:                        # noqa: BLE001
        rows.append((
            BAD, "device",
            f"{type(exc).__name__}: {exc}\n"
            f"          No Vulkan 1.2 device. Update your GPU driver; on Linux also "
            f"install the loader (`libvulkan1` on Debian/Ubuntu, `vulkan-loader` on "
            f"Fedora/Arch). `vulkaninfo --summary` is the next thing to try.",
        ))
        return rows

    # a real dispatch, because every check above can pass on a device that
    # cannot actually run a kernel
    try:
        x = grilly.ones(4) + grilly.ones(4)
        got = x.numpy().tolist()
        status = OK if got == [2.0, 2.0, 2.0, 2.0] else BAD
        rows.append((status, "a kernel ran", f"ones(4) + ones(4) = {got}"))
    except Exception as exc:                        # noqa: BLE001
        rows.append((BAD, "a kernel ran", f"{type(exc).__name__}: {exc}"))

    return rows


def report() -> list:
    """The diagnostic rows. Separate from printing so a test can read it."""
    return _rows()


def main(argv=None) -> int:
    rows = report()
    width = max(len(label) for _, label, _ in rows)
    mark = {OK: "[ok]  ", BAD: "[FAIL]", MEH: "[warn]"}
    for status, label, detail in rows:
        print(f"{mark[status]} {label.ljust(width)}  {detail}")
    bad = sum(1 for status, _, _ in rows if status == BAD)
    print()
    if bad:
        print(f"{bad} problem(s). Please include everything above in a bug report:")
        print("  https://github.com/Grillcheese-AI/grilly2/issues")
        return 1
    print("grilly is working on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
