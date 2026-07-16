"""TONE3000 API client: search the catalog without downloading, then fetch
only shortlisted .nam captures.

Auth: OAuth 2.0 + PKCE against your TONE3000 account, using a publishable API
key (tone3000.com -> Settings -> API Keys, `t3k_pub_...`). Localhost redirect
URIs are allowed without registration. Tokens are cached in ~/.tonematch/.

CLI:
    python -m tonematch.tone3000 connect --key t3k_pub_...
    python -m tonematch.tone3000 search "plexi" --gear amp --sort trending
    python -m tonematch.tone3000 download 12345 67890 --out ./t3k_cache
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
import webbrowser

import requests

T3K_API = "https://www.tone3000.com"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".tonematch", "tone3000.json")

# Module-level lock around the shared token config. Multiple threads (e.g. one
# per JobManager worker) refresh tokens through the same client singleton; this
# keeps two refreshes from racing on the same expiring token.
_CONFIG_LOCK = threading.Lock()

GEARS = ("amp", "full-rig", "pedal", "outboard", "ir")
SORTS = ("best-match", "newest", "oldest", "trending", "downloads-all-time")


class T3KError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Config / token storage
# ----------------------------------------------------------------------------


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    """Atomically replace the TONE3000 config file.

    `os.replace` is atomic on both POSIX and Windows, so half-written config
    can't be observed by another process/thread even if we crash mid-write.
    """
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    tmp = CONFIG_PATH + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2)
    os.replace(tmp, CONFIG_PATH)


# ----------------------------------------------------------------------------
# OAuth 2.0 PKCE with a local callback listener
# ----------------------------------------------------------------------------


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _CallbackHandler.result = {k: v[0] for k, v in q.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:sans-serif'><h2>NAM EQ Matcher connected to "
            b"TONE3000 &#127928;</h2>You can close this tab.</body></html>"
        )

    def log_message(self, *args):  # silence
        pass


def login(publishable_key: str, timeout_s: int = 180, open_browser: bool = True) -> dict:
    """Run the PKCE flow. Opens the browser, waits for the callback, stores tokens."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(16)

    server = http.server.HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    port = server.server_address[1]
    redirect_uri = f"http://localhost:{port}"
    _CallbackHandler.result = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    params = {
        "client_id": publishable_key,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    url = f"{T3K_API}/api/v1/oauth/authorize?{urllib.parse.urlencode(params)}"
    if open_browser:
        webbrowser.open(url)
    print(f"If the browser didn't open, visit:\n{url}")

    t0 = time.time()
    try:
        while not _CallbackHandler.result:
            if time.time() - t0 > timeout_s:
                raise T3KError("Timed out waiting for TONE3000 login.")
            time.sleep(0.25)
    finally:
        server.shutdown()

    res = _CallbackHandler.result
    if res.get("state") != state:
        raise T3KError("OAuth state mismatch - try again.")
    if "error" in res:
        raise T3KError(f"TONE3000 auth error: {res['error']}")
    if "code" not in res:
        raise T3KError("No authorization code received.")

    r = requests.post(
        f"{T3K_API}/api/v1/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": res["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": publishable_key,
        },
        timeout=30,
    )
    if not r.ok:
        raise T3KError(f"Token exchange failed: {r.status_code} {r.text[:300]}")
    data = r.json()
    tokens = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + data["expires_in"],
    }
    cfg = _load_config()
    cfg["publishable_key"] = publishable_key
    cfg["tokens"] = tokens
    _save_config(cfg)
    return tokens


# ----------------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------------


class T3KClient:
    def __init__(self, publishable_key: str | None = None):
        cfg = _load_config()
        self.publishable_key = publishable_key or cfg.get("publishable_key")
        self.tokens = cfg.get("tokens")

    @property
    def connected(self) -> bool:
        return bool(self.tokens)

    def connect(self, publishable_key: str | None = None) -> None:
        key = publishable_key or self.publishable_key
        if not key:
            raise T3KError(
                "No publishable key. Create one at tone3000.com -> Settings -> API Keys."
            )
        self.publishable_key = key
        self.tokens = login(key)

    def _access_token(self) -> str:
        if not self.tokens:
            raise T3KError("Not connected to TONE3000. Run connect first.")
        if time.time() > self.tokens["expires_at"] - 60:
            with _CONFIG_LOCK:
                # Re-check inside the lock; another thread may have refreshed
                # while we were waiting.
                if not self.tokens or time.time() <= self.tokens["expires_at"] - 60:
                    return self.tokens["access_token"]
                r = requests.post(
                    f"{T3K_API}/api/v1/oauth/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.tokens["refresh_token"],
                        "client_id": self.publishable_key,
                    },
                    timeout=30,
                )
                if not r.ok:
                    self.tokens = None
                    cfg = _load_config()
                    cfg.pop("tokens", None)
                    _save_config(cfg)
                    raise T3KError("Session expired - please connect to TONE3000 again.")
                data = r.json()
                self.tokens = {
                    "access_token": data["access_token"],
                    "refresh_token": data["refresh_token"],
                    "expires_at": time.time() + data["expires_in"],
                }
                cfg = _load_config()
                cfg["tokens"] = self.tokens
                _save_config(cfg)
        return self.tokens["access_token"]

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = requests.get(
            f"{T3K_API}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._access_token()}"},
            timeout=60,
        )
        if r.status_code == 429:
            raise T3KError("TONE3000 rate limit hit (100 req/min) - wait a minute.")
        if not r.ok:
            raise T3KError(f"GET {path} failed: {r.status_code} {r.text[:300]}")
        return r.json()

    # -- catalog ---------------------------------------------------------------

    def search_tones(
        self,
        query: str = "",
        gear: str | None = None,
        sort: str = "downloads-all-time",
        page: int = 1,
        page_size: int = 30,
        nam_only: bool = True,
    ) -> list[dict]:
        """Search the catalog. Returns tone metadata only - no files."""
        params: dict = {"page": page, "page_size": page_size, "sort": sort}
        if query:
            params["query"] = query
        if gear and gear != "any":
            params["gears"] = gear
        data = self._get("/api/v1/tones/search", params)
        tones = data.get("data", [])
        if nam_only:
            tones = [t for t in tones if t.get("format") == "nam"]
        return tones

    def search_tones_refined(
        self,
        query: str = "",
        amps: list[str] | None = None,
        gear: str | None = None,
        sort: str = "downloads-all-time",
        page_size: int = 30,
    ) -> list[dict]:
        """Search with amp/make refinement.

        `amps` is a list of amp names/makes/model nicknames (e.g. ["Marshall",
        "JCM800", "5150"]). One catalog query is run per amp term (combined
        with `query`), results are merged and de-duplicated, then re-ranked by
        where the term matched: the tone's `makes` metadata ranks highest,
        then title, then tags. Each returned tone gets a `_matched_on` note.
        """
        amps = [a.strip() for a in (amps or []) if a.strip()][:6]  # rate-limit friendly
        if not amps:
            return self.search_tones(query, gear=gear, sort=sort, page_size=page_size)

        merged: dict[int, dict] = {}
        for amp in amps:
            q = f"{query} {amp}".strip()
            for t in self.search_tones(q, gear=gear, sort=sort, page_size=page_size):
                merged.setdefault(t["id"], t)

        def match_info(tone: dict) -> tuple[float, str]:
            makes = " ".join(m["name"] for m in tone.get("makes", [])).lower()
            title = (tone.get("title") or "").lower()
            tags = " ".join(g["name"] for g in tone.get("tags", [])).lower()
            desc = (tone.get("description") or "").lower()
            score, hits = 0.0, []
            for amp in amps:
                a = amp.lower()
                if a in makes:
                    score += 3.0
                    hits.append(f"make:{amp}")
                elif a in title:
                    score += 2.0
                    hits.append(f"title:{amp}")
                elif a in tags:
                    score += 1.0
                    hits.append(f"tag:{amp}")
                elif a in desc:
                    score += 0.5
                    hits.append(f"desc:{amp}")
            return score, ", ".join(hits) or "-"

        ranked = []
        for t in merged.values():
            score, hits = match_info(t)
            t["_matched_on"] = hits
            t["_match_score"] = score
            ranked.append(t)
        # amp-metadata matches first, then popularity
        ranked.sort(key=lambda t: (-t["_match_score"], -t.get("downloads_count", 0)))
        return ranked[:page_size]

    def list_models(self, tone_id: int) -> list[dict]:
        data = self._get("/api/v1/models", {"tone_id": tone_id, "page_size": 100})
        return data.get("data", [])

    def download_model(self, model: dict, dest_dir: str) -> str:
        """Download one model file (Bearer-authenticated). Returns local path."""
        os.makedirs(dest_dir, exist_ok=True)
        url = model["model_url"]
        storage_name = urllib.parse.urlparse(url).path.split("/")[-1]
        ext = "." + storage_name.split(".")[-1] if "." in storage_name else ".nam"
        safe = re.sub(r"[^a-z0-9]+", "-", model["name"].lower()).strip("-") or f"model-{model['id']}"
        out_path = os.path.join(dest_dir, f"{safe}-{model['id']}{ext}")
        if os.path.exists(out_path):
            return out_path
        path = url.replace(T3K_API, "")
        r = requests.get(
            f"{T3K_API}{path}" if path.startswith("/") else url,
            headers={"Authorization": f"Bearer {self._access_token()}"},
            timeout=120,
        )
        if not r.ok:
            raise T3KError(f"Download failed ({r.status_code}) for {model['name']}")
        with open(out_path, "wb") as fp:
            fp.write(r.content)
        return out_path

    def download_tone(self, tone_id: int, dest_dir: str, prefer_size: str = "standard") -> list[str]:
        """Download a tone's .nam models (preferring one size class). Returns paths."""
        models = [m for m in self.list_models(tone_id) if m["model_url"].lower().endswith(".nam")]
        if not models:
            return []
        preferred = [m for m in models if m.get("size") == prefer_size]
        chosen = preferred or models
        return [self.download_model(m, dest_dir) for m in chosen]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Search/download NAM captures from TONE3000")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_connect = sub.add_parser("connect", help="Log in to TONE3000")
    p_connect.add_argument("--key", required=True, help="Publishable key (t3k_pub_...)")

    p_search = sub.add_parser("search", help="Search the catalog (metadata only)")
    p_search.add_argument("query", nargs="?", default="")
    p_search.add_argument("--amps", default="", help='Comma-separated amp names, e.g. "Marshall,JCM800,5150"')
    p_search.add_argument("--gear", default="any", choices=("any",) + GEARS)
    p_search.add_argument("--sort", default="downloads-all-time", choices=SORTS)
    p_search.add_argument("--n", type=int, default=20)

    p_dl = sub.add_parser("download", help="Download tones' .nam models by tone ID")
    p_dl.add_argument("tone_ids", nargs="+", type=int)
    p_dl.add_argument("--out", default="t3k_cache")

    args = ap.parse_args()
    client = T3KClient()

    if args.cmd == "connect":
        client.connect(args.key)
        print("Connected. Tokens stored in", CONFIG_PATH)
    elif args.cmd == "search":
        amps = [a for a in args.amps.split(",") if a.strip()]
        tones = client.search_tones_refined(
            args.query, amps=amps, gear=args.gear, sort=args.sort, page_size=args.n
        )
        for t in tones:
            makes = ", ".join(m["name"] for m in t.get("makes", []))
            matched = f"  [{t['_matched_on']}]" if t.get("_matched_on") else ""
            print(
                f"  [{t['id']:>6}] {t['title'][:48]:48s}  {t['gear']:8s} "
                f"{t['models_count']:3d} models  {t['downloads_count']:6d} dl  {makes}{matched}"
            )
        if not tones:
            print("No NAM tones found.")
    elif args.cmd == "download":
        for tid in args.tone_ids:
            paths = client.download_tone(tid, args.out)
            for p in paths:
                print("downloaded:", p)
        print("Done. Point --models at", os.path.abspath(args.out))


if __name__ == "__main__":
    _main()
