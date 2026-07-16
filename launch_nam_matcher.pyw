"""Silent launcher for NAM EQ Matcher (app.py).

Double-click this file (with pythonw.exe associated to .pyw) to launch the
Gradio app with no console window. It opens your browser automatically and
shuts the server down shortly after you actually close the tab (switching
tabs or minimizing the window will NOT trigger a shutdown).

Place this file in the SAME FOLDER as app.py.
"""

import os
import sys
import time
import socket
import threading
import traceback
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "launcher_error.log")

# Make sure app.py (and any sibling "tonematch" package) is importable, and
# that any relative paths app.py uses (e.g. "t3k_cache") land next to it.
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)


class _NullStream:
    """Stand-in for stdout/stderr under pythonw.exe, where both are None.

    Several libraries (uvicorn's logging setup among them) unconditionally
    call methods like isatty()/write()/flush() on sys.stdout/sys.stderr,
    which crashes with AttributeError when those are None (as they are with
    no console attached). This just swallows everything safely.
    """

    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def isatty(self):
        return False


if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()

PORT = 7860
LAST_HEARTBEAT = time.time()
CLOSE_REQUESTED_AT = None  # set when the page signals it's actually closing
STATE_LOCK = threading.Lock()

# How long to wait after a "closing" signal before actually shutting down.
# A plain page refresh also fires the closing signal, but is immediately
# followed by a fresh heartbeat from the reloaded page -- which cancels the
# pending shutdown. Only a real close goes quiet for this long.
CLOSE_GRACE_SECONDS = 8.0

# Safety net only: if the page never gets a chance to signal closing at all
# (browser crash, force-quit, OS killing the process), fall back to shutting
# down after this much total silence. Kept long and NOT used for ordinary
# "tab in the background" situations -- browsers throttle timers in
# backgrounded tabs, so a short timeout here would shut the server down just
# because you switched tabs, not because you closed anything.
STALE_FALLBACK_SECONDS = 30 * 60

# Injected into the page <head>.
#  - Regular heartbeat pings while the tab is around, so we know it's alive.
#  - A dedicated "closing" beacon on pagehide, which is what actually decides
#    shutdown -- this fires on real close/navigation, not on merely
#    switching to another tab or minimizing the window.
HEARTBEAT_SNIPPET = """
<script>
(function () {
    function heartbeat() {
        fetch('/heartbeat', { method: 'GET', cache: 'no-store', keepalive: true }).catch(function () {});
    }
    heartbeat();
    setInterval(heartbeat, 15000);

    window.addEventListener('pagehide', function () {
        // sendBeacon is the reliable way to get a request out during unload;
        // fires on real tab close AND on refresh/navigation (server tells
        // those apart by whether a new heartbeat follows shortly after).
        if (navigator.sendBeacon) {
            navigator.sendBeacon('/closing');
        }
    });
})();
</script>
"""


def log_error(exc: BaseException) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n--- %s ---\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
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


def wait_for_server_and_open_browser(port: int, timeout: float = 30.0) -> None:
    """Opens the browser the instant the server actually accepts connections,
    instead of guessing a fixed delay (which either wastes time or opens the
    tab too early and shows a connection error)."""
    url = "http://127.0.0.1:%d" % port
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.15)
    webbrowser.open(url)  # fall back and try anyway if it's taking unusually long


def build_app():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from app import demo  # imports app.py; its own __main__ block does not run

    # Inject the heartbeat script into the Gradio page's <head>.
    existing_head = getattr(demo, "head", None) or ""
    demo.head = existing_head + HEARTBEAT_SNIPPET

    fastapi_app = FastAPI()

    @fastapi_app.get("/heartbeat")
    def heartbeat():
        global LAST_HEARTBEAT, CLOSE_REQUESTED_AT
        with STATE_LOCK:
            LAST_HEARTBEAT = time.time()
            CLOSE_REQUESTED_AT = None  # a fresh heartbeat means the page is still around (e.g. just refreshed)
        return JSONResponse({"ok": True})

    @fastapi_app.post("/closing")
    def closing():
        global CLOSE_REQUESTED_AT
        with STATE_LOCK:
            CLOSE_REQUESTED_AT = time.time()
        return JSONResponse({"ok": True})

    gr.mount_gradio_app(fastapi_app, demo, path="/")
    return fastapi_app


def monitor_shutdown():
    """Shuts the process down on an actual tab close, not on backgrounding."""
    time.sleep(45)  # grace period for startup / first page load
    while True:
        time.sleep(1)
        with STATE_LOCK:
            close_at = CLOSE_REQUESTED_AT
            last_heartbeat = LAST_HEARTBEAT
        now = time.time()

        # A real close (not a refresh) -- the closing signal arrived and no
        # follow-up heartbeat cancelled it within the grace window.
        if close_at is not None and (now - close_at) > CLOSE_GRACE_SECONDS:
            os._exit(0)

        # Long-tail safety net: pagehide never fired at all (crash / force
        # quit / OS killed the process). Intentionally long so ordinary
        # backgrounded/throttled tabs are never mistaken for this.
        if (now - last_heartbeat) > STALE_FALLBACK_SECONDS:
            os._exit(0)


def main():
    global LAST_HEARTBEAT, PORT
    LAST_HEARTBEAT = time.time()

    import uvicorn

    PORT = find_free_port(PORT)
    fastapi_app = build_app()

    threading.Thread(target=monitor_shutdown, daemon=True).start()

    threading.Thread(
        target=wait_for_server_and_open_browser, args=(PORT,), daemon=True
    ).start()

    uvicorn.run(
        fastapi_app,
        host="127.0.0.1",
        port=PORT,
        log_level="warning",
        log_config=None,  # skip uvicorn's dictConfig entirely; avoids touching stdout/stderr internals
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # no console to see this in .pyw mode
        log_error(exc)
        os._exit(1)
