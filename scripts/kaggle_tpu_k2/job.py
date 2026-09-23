"""Talk to a running runner notebook.

    python job.py send path/to/script.py|.sh   # signed job -> IN topic
    python job.py stop                          # stop the job loop
    python job.py tail [since]                  # print OUT topic messages (+ fetch logs)
"""
import hashlib
import hmac
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / ".state.json").read_text())
LOGS = HERE / "build" / "logs"


def put(topic, body, headers):
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body, headers=headers,
                                 method="PUT")
    print(urllib.request.urlopen(req, timeout=60).read().decode()[:200])


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
    raw = urllib.request.urlopen(
        f"https://ntfy.sh/{S['out_topic']}/json?poll=1&since={since}", timeout=60).read()
    LOGS.mkdir(parents=True, exist_ok=True)
    for ln in raw.decode().splitlines():
        m = json.loads(ln)
        if m.get("event") != "message":
            continue
        att = m.get("attachment")
        if att:
            dest = LOGS / f"{m['id']}_{att['name']}"
            if not dest.exists():
                dest.write_bytes(urllib.request.urlopen(att["url"], timeout=120).read())
            print(f"{m['id']} [file] {m.get('title', '')} -> {dest}")
        else:
            print(f"{m['id']} {m.get('message', '')}")
