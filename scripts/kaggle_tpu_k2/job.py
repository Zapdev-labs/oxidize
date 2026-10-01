"""Talk to a running runner notebook.

    python job.py send path/to/script.py|.sh   # signed job -> IN topic
    python job.py stop                          # stop the job loop
    python job.py tail [since]                  # print OUT topic messages (+ fetch logs)
"""
import hashlib
import hmac
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / ".state.json").read_text())
LOGS = HERE / "build" / "logs"
NTFY = "https://ntfy.sh"
NTFY_HOST = urllib.parse.urlsplit(NTFY).hostname


def _allowed_url(url):
    u = urllib.parse.urlsplit(url)
    return u.scheme == "https" and u.hostname == NTFY_HOST


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _allowed_url(newurl):
            raise urllib.error.HTTPError(newurl, code, f"refusing redirect to {newurl}",
                                         headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_HttpsOnlyRedirect)


def fetch_attachment(url, timeout=120):
    if not _allowed_url(url):
        raise ValueError(f"attachment URL not https://{NTFY_HOST}: {url[:200]}")
    with _opener.open(url, timeout=timeout) as r:
        return r.read()


def put(topic, body, headers):
    req = urllib.request.Request(f"{NTFY}/{topic}", data=body, headers=headers,
                                 method="PUT")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(r.read().decode()[:200])


cmd = sys.argv[1]
if cmd == "send":
    p = Path(sys.argv[2])
    body = p.read_bytes()
    sig = hmac.new(S["hmac_key"].encode(), body, hashlib.sha256).hexdigest()
    put(S["in_topic"], body, {"Filename": p.name, "Title": sig})
elif cmd == "stop":
    put(S["in_topic"], b"stop", {})
elif cmd == "tail":
    since = sys.argv[2] if len(sys.argv) > 2 else "all"
    with urllib.request.urlopen(
            f"{NTFY}/{S['out_topic']}/json?poll=1&since={since}", timeout=60) as r:
        raw = r.read()
    LOGS.mkdir(parents=True, exist_ok=True)
    logs_root = LOGS.resolve()
    for ln in raw.decode(errors="replace").splitlines():
        try:
            m = json.loads(ln)
        except ValueError:
            print(f"(skipping malformed line: {ln[:200]!r})", file=sys.stderr)
            continue
        if not isinstance(m, dict) or m.get("event") != "message":
            continue
        mid = Path(str(m.get("id", ""))).name
        att = m.get("attachment")
        if isinstance(att, dict) and att.get("url"):
            name = Path(str(att.get("name") or "attachment")).name or "attachment"
            dest = LOGS / f"{mid}_{name}"
            if dest.resolve().parent != logs_root:
                print(f"{mid} (skipping attachment with bad name {name!r})", file=sys.stderr)
                continue
            try:
                if not dest.exists():
                    dest.write_bytes(fetch_attachment(str(att["url"])))
            except Exception as e:  # noqa: BLE001
                print(f"{mid} (attachment download failed: {e!r})", file=sys.stderr)
                continue
            print(f"{mid} [file] {m.get('title', '')} -> {dest}")
        else:
            print(f"{mid} {m.get('message', '')}")
