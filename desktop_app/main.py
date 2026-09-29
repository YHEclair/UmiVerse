from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_STDIO_HANDLES = []


def configure_frozen_stdio() -> None:
    if not getattr(sys, "frozen", False):
        return
    for stream_name, mode in (("stdout", "w"), ("stderr", "w"), ("stdin", "r")):
        if getattr(sys, stream_name) is not None:
            continue
        handle = open(os.devnull, mode, encoding="utf-8")
        _STDIO_HANDLES.append(handle)
        setattr(sys, stream_name, handle)


def configure_frozen_qt_paths() -> None:
    if not getattr(sys, "frozen", False):
        return
    exe_dir = Path(sys.executable).resolve().parent
    internal_dir = Path(getattr(sys, "_MEIPASS", exe_dir))
    candidates = [
        internal_dir / "PySide6",
        internal_dir / "PySide6" / "Qt6" / "bin",
        internal_dir / "PySide6" / "plugins",
        internal_dir / "PySide6" / "plugins" / "platforms",
        internal_dir / "shiboken6",
        internal_dir,
        exe_dir,
    ]
    dll_names = {
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "shiboken6.abi3.dll",
        "qwindows.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "MSVCP140.dll",
        "icuuc*.dll",
        "icuin*.dll",
        "icudt*.dll",
    }
    for dll_name in dll_names:
        candidates.extend(path.parent for path in internal_dir.rglob(dll_name))

    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        os.environ["PATH"] = f"{resolved}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved))
    plugin_dir = internal_dir / "PySide6" / "plugins"
    if plugin_dir.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_dir))
        platform_dir = plugin_dir / "platforms"
        if platform_dir.exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platform_dir))
    if os.environ.get("UMIVERSE_DLL_DEBUG") == "1" or Path(sys.executable).stem.endswith("_Debug"):
        print_frozen_dll_diagnostics(exe_dir, internal_dir)


def print_frozen_dll_diagnostics(exe_dir: Path, internal_dir: Path) -> None:
    print("=== UmiVerse DLL diagnostics ===", flush=True)
    print(f"executable: {sys.executable}", flush=True)
    print(f"exe_dir: {exe_dir}", flush=True)
    print(f"_MEIPASS/internal_dir: {internal_dir}", flush=True)
    print(f"QT_PLUGIN_PATH: {os.environ.get('QT_PLUGIN_PATH', '')}", flush=True)
    print(
        f"QT_QPA_PLATFORM_PLUGIN_PATH: {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', '')}",
        flush=True,
    )
    targets = [
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "shiboken6.abi3.dll",
        "qwindows.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "MSVCP140.dll",
        "libcrypto-3-x64.dll",
        "libssl-3-x64.dll",
        "zlib.dll",
        "icuuc*.dll",
        "icuin*.dll",
        "icudt*.dll",
    ]
    for pattern in targets:
        matches = sorted(internal_dir.rglob(pattern))
        if matches:
            print(f"{pattern}: {len(matches)} found", flush=True)
            for match in matches[:8]:
                print(f"  {match}", flush=True)
        else:
            print(f"{pattern}: NOT FOUND", flush=True)
    print("ctypes load attempts:", flush=True)
    for name in ["Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "shiboken6.abi3.dll"]:
        try:
            ctypes.WinDLL(name)
            print(f"  OK {name}", flush=True)
        except OSError as exc:
            print(f"  FAIL {name}: {exc}", flush=True)
    print("=== end diagnostics ===", flush=True)


configure_frozen_stdio()
configure_frozen_qt_paths()

from PySide6.QtWidgets import QApplication

from umiverse_desktop import APP_NAME, __version__
from umiverse_desktop.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    window = MainWindow(project_root=Path(__file__).resolve().parents[1])
    window.resize(1280, 820)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
