"""Standalone desktop launcher for NAM EQ Matcher (pywebview).

Double-click (with pythonw.exe associated to .pyw) or run:
    python launch_nam_matcher_app.pyw

Opens the Gradio app inside its own OS window. Unlike the browser-tab
launcher (launch_nam_matcher.pyw), this one:
  - Cannot be accidentally closed with Ctrl-W or by closing the browser.
  - Prompts for confirmation only if you close the window while a match
    run is in progress (the root-cause fix for losing long runs).
  - Needs no heartbeat/pagehide heuristics; the window lifetime IS the
    app lifetime.

Requires `pywebview` (in requirements.txt). On Windows, uses Edge WebView2,
already present on Windows 10/11 via the Edge browser.

Place this file in the SAME FOLDER as app.py.
"""

import os
import sys
import time
import socket
import threading
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "launcher_app_error.log")
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)


class _NullStream:
    def write(self, *a, **kw): pass
    def flush(self, *a, **kw): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()

PORT = 7860
HOST = "127.0.0.1"
APP_TITLE = "NAM EQ Matcher"
_match_code = None  # set in build_fastapi_app after importing app


def is_run_in_progress() -> bool:
    """Check live via thread stacks whether a tone match is executing."""
    if _match_code is None:
        return False
    for _tid, frame in sys._current_frames().items():
        f = frame
        while f:
            if f.f_code is _match_code:
                return True
            f = f.f_back
    return False



def log_error(exc: BaseException) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n--- %s ---\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    except Exception:
        pass


def _show_error(title: str, message: str) -> None:
    """Best-effort popup; tkinter is always present on Windows Python."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass


def find_free_port(start_port: int, attempts: int = 15) -> int:
    port = start_port
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.15)
    log_error(Exception("wait_for_port timed out after %.1fs on port %d" % (timeout, port)))
    return False


_BEFOREUNLOAD_SNIPPET = """
<script>
(function () {
    window.addEventListener('beforeunload', function (e) {
        try {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/run-status', false);
            xhr.send(null);
            if (xhr.status === 200) {
                var d = JSON.parse(xhr.responseText);
                if (d && d.busy) {
                    e.preventDefault();
                    e.returnValue = '';
                }
            } else {
                e.preventDefault();
                e.returnValue = '';
            }
        } catch (err) {
            e.preventDefault();
            e.returnValue = '';
        }
    });
})();
</script>
"""


_LOADING_HTML = """<!DOCTYPE html>
<html><head><style>
body { margin:0; display:flex; align-items:center; justify-content:center;
       height:100vh; background:#1a1a2e; color:#e0e0e0; font-family:sans-serif; }
.spinner { width:36px; height:36px; border:4px solid #444; border-top-color:#7c8aff;
           border-radius:50%; animation:spin .8s linear infinite; margin-right:16px; }
@keyframes spin { to { transform:rotate(360deg); } }
</style></head><body>
<div class="spinner"></div>
<div><strong>NAM EQ Matcher</strong><br><span style="color:#999">Starting server&hellip;</span></div>
</body></html>
"""


def build_fastapi_app():
    global _match_code
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import app as app_module
    from app import demo
    from tonematch.job_manager import _run_match_core
    _match_code = _run_match_core.__code__
    existing_head = getattr(demo, "head", None) or ""
    demo.head = existing_head + _BEFOREUNLOAD_SNIPPET
    fastapi_app = FastAPI()

    @fastapi_app.get("/run-status")
    def run_status():
        return JSONResponse({"busy": is_run_in_progress()})

    gr.mount_gradio_app(fastapi_app, demo, path="/")
    return fastapi_app


def run_uvicorn(fastapi_app_obj, port: int) -> None:
    import uvicorn
    uvicorn.run(
        fastapi_app_obj,
        host=HOST,
        port=port,
        log_level="warning",
        log_config=None,  # skip uvicorn's dictConfig entirely; avoids touching stdout/stderr internals
    )


def main():
    try:
        import webview
    except ImportError as exc:
        log_error(exc)
        _show_error(
            "Missing dependency",
            "pywebview is required but not installed.\n\n"
            "Fix — run this in a terminal:\n\n"
            "    pip install pywebview\n\n"
            "Then double-click this file again."
        )
        os._exit(2)

    port = find_free_port(PORT)
    fastapi_app = build_fastapi_app()
    threading.Thread(target=run_uvicorn, args=(fastapi_app, port), daemon=True).start()

    window = webview.create_window(
        APP_TITLE, html=_LOADING_HTML,
        width=1400, height=1000, min_size=(1000, 700),
    )

    def _on_closing():
        if is_run_in_progress():
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                result = messagebox.askyesno(
                    "NAM EQ Matcher",
                    "A tone match is still running.\n\n"
                    "Closing now will cancel the current job.\n\n"
                    "Do you want to close anyway?"
                )
                root.destroy()
                return result
            except Exception:
                return True
        return True

    window.events.closing += _on_closing

    def _after_start():
        wait_for_port(port, timeout=30.0)
        window.load_url(f"http://{HOST}:{port}")

    webview.start(func=_after_start)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_error(exc)
        _show_error(
            "NAM EQ Matcher — startup error",
            f"{type(exc).__name__}: {exc}\n\n"
            f"Full traceback written to:\n{LOG_PATH}\n\n"
            "If a module is missing, run:\n"
            "    pip install -r requirements.txt"
        )
        os._exit(1)
