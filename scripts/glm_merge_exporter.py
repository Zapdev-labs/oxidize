#!/usr/bin/env python3
"""Prometheus exporter for GLM-5.1 / GLM-5.2 oxidize stream-merge pipeline + host stats."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PHASE_IDLE = 0
PHASE_DOWNLOAD = 1
PHASE_MERGE = 2
PHASE_COMPLETE = 3

SHARD_TOTAL = 282
TARGET_BYTES = 1_508_000_000_000  # ~1.5 TB per model

_last_cpu_sample: tuple[int, int] | None = None
_dir_bytes_cache: dict[str, tuple[float, int]] = {}
_DIR_BYTES_TTL_SEC = 60.0

def proc_root() -> Path:
    host = Path(os.environ.get("HOST_PROC", "/host/proc"))
    return host if host.exists() else Path("/proc")


def sysfs_root() -> Path:
    host = Path(os.environ.get("HOST_SYS", "/host/sys"))
    return host if host.exists() else Path("/sys")


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def proc_running(*patterns: str) -> bool:
    proc = proc_root()
    try:
        for pid_dir in proc.iterdir():
            if not pid_dir.is_dir() or not pid_dir.name.isdigit():
                continue
            cmdline = pid_dir / "cmdline"
            if not cmdline.exists():
                continue
            cmd = cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            if any(p in cmd for p in patterns):
                return True
    except OSError:
        pass
    return any(run(["pgrep", "-f", p]) != "" for p in patterns)


def read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in (proc_root() / "meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts:
                out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def read_loadavg() -> tuple[float, float, float]:
    try:
        parts = (proc_root() / "loadavg").read_text().split()[:3]
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, IndexError, ValueError):
        return 0.0, 0.0, 0.0


def read_uptime() -> float:
    try:
        return float((proc_root() / "uptime").read_text().split()[0])
    except (OSError, ValueError):
        return 0.0


def read_cpu_percent() -> float:
    global _last_cpu_sample
    try:
        cpu_line = (proc_root() / "stat").read_text().splitlines()[0]
        parts = [int(x) for x in cpu_line.split()[1:]]
        idle = parts[3] + parts[4]
        total = sum(parts)
    except (OSError, IndexError, ValueError):
        return 0.0
    if _last_cpu_sample is None:
        _last_cpu_sample = (total, idle)
        return 0.0
    prev_total, prev_idle = _last_cpu_sample
    _last_cpu_sample = (total, idle)
    dt = total - prev_total
    didle = idle - prev_idle
    if dt <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - didle / dt)))


def read_network_counters() -> list[tuple[str, int, int, int, int]]:
    """iface, rx_bytes, tx_bytes, rx_packets, tx_packets."""
    rows: list[tuple[str, int, int, int, int]] = []
    try:
        lines = (proc_root() / "net" / "dev").read_text().splitlines()[2:]
        for line in lines:
            if ":" not in line:
                continue
            iface, data = line.split(":", 1)
            iface = iface.strip()
            if iface == "lo" or iface.startswith("veth") or iface.startswith("docker"):
                continue
            cols = data.split()
            if len(cols) < 16:
                continue
            rows.append(
                (
                    iface,
                    int(cols[0]),
                    int(cols[8]),
                    int(cols[1]),
                    int(cols[9]),
                )
            )
    except OSError:
        pass
    return rows


def read_diskstats() -> dict[str, tuple[int, int, int, int]]:
    """device -> (read_sectors, write_sectors, read_ios, write_ios)."""
    out: dict[str, tuple[int, int, int, int]] = {}
    try:
        for line in (proc_root() / "diskstats").read_text().splitlines():
            cols = line.split()
            if len(cols) < 14:
                continue
            dev = cols[2]
            out[dev] = (int(cols[5]), int(cols[9]), int(cols[3]), int(cols[7]))
    except OSError:
        pass
    return out


def home_block_device() -> str:
    try:
        mounts = (proc_root() / "mounts").read_text().splitlines()
        for line in mounts:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "/home":
                dev = Path(parts[0]).name
                if dev.startswith("dm-"):
                    slink = sysfs_root() / "block" / dev / "slaves"
                    if slink.is_dir():
                        slaves = sorted(p.name for p in slink.iterdir())
                        if slaves:
                            return slaves[0]
                return dev
    except OSError:
        pass
    return "sda"


def host_metrics_lines() -> list[str]:
    mem = read_meminfo()
    mem_total = mem.get("MemTotal", 0)
    mem_free = mem.get("MemFree", 0)
    mem_avail = mem.get("MemAvailable", mem_free)
    mem_used = max(0, mem_total - mem_avail)
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)
    load1, load5, load15 = read_loadavg()
    cpu_pct = read_cpu_percent()
    uptime = read_uptime()

    lines = [
        "# HELP host_cpu_usage_percent Host CPU busy percent (sampled between scrapes).",
        "# TYPE host_cpu_usage_percent gauge",
        f"host_cpu_usage_percent {cpu_pct:.4f}",
        "# HELP host_load_average Host load average.",
        "# TYPE host_load_average gauge",
        f'host_load_average{{period="1m"}} {load1:.4f}',
        f'host_load_average{{period="5m"}} {load5:.4f}',
        f'host_load_average{{period="15m"}} {load15:.4f}',
        "# HELP host_uptime_seconds Host uptime.",
        "# TYPE host_uptime_seconds gauge",
        f"host_uptime_seconds {uptime:.2f}",
        "# HELP host_memory_bytes Host memory bytes.",
        "# TYPE host_memory_bytes gauge",
        f'host_memory_bytes{{kind="total"}} {mem_total}',
        f'host_memory_bytes{{kind="used"}} {mem_used}',
        f'host_memory_bytes{{kind="available"}} {mem_avail}',
        f'host_memory_bytes{{kind="free"}} {mem_free}',
        "# HELP host_memory_used_percent Host memory used percent.",
        "# TYPE host_memory_used_percent gauge",
        f"host_memory_used_percent {(100.0 * mem_used / mem_total) if mem_total else 0.0:.4f}",
        "# HELP host_swap_bytes Host swap bytes.",
        "# TYPE host_swap_bytes gauge",
        f'host_swap_bytes{{kind="total"}} {swap_total}',
        f'host_swap_bytes{{kind="used"}} {swap_used}',
    ]

    net = read_network_counters()
    if net:
        lines.extend(
            [
                "# HELP host_network_receive_bytes_total Network bytes received.",
                "# TYPE host_network_receive_bytes_total counter",
                "# HELP host_network_transmit_bytes_total Network bytes transmitted.",
                "# TYPE host_network_transmit_bytes_total counter",
                "# HELP host_network_receive_packets_total Network packets received.",
                "# TYPE host_network_receive_packets_total counter",
                "# HELP host_network_transmit_packets_total Network packets transmitted.",
                "# TYPE host_network_transmit_packets_total counter",
            ]
        )
        for iface, rx_b, tx_b, rx_p, tx_p in net:
            lines.extend(
                [
                    f'host_network_receive_bytes_total{{device="{iface}"}} {rx_b}',
                    f'host_network_transmit_bytes_total{{device="{iface}"}} {tx_b}',
                    f'host_network_receive_packets_total{{device="{iface}"}} {rx_p}',
                    f'host_network_transmit_packets_total{{device="{iface}"}} {tx_p}',
                ]
            )

    dev = home_block_device()
    stats = read_diskstats()
    if dev in stats:
        rd_sectors, wr_sectors, rd_ios, wr_ios = stats[dev]
        lines.extend(
            [
                "# HELP host_disk_io_bytes_total Block device IO bytes (512-byte sectors).",
                "# TYPE host_disk_io_bytes_total counter",
                f'host_disk_io_bytes_total{{device="{dev}",op="read"}} {rd_sectors * 512}',
                f'host_disk_io_bytes_total{{device="{dev}",op="write"}} {wr_sectors * 512}',
                "# HELP host_disk_io_ops_total Block device IO operations.",
                "# TYPE host_disk_io_ops_total counter",
                f'host_disk_io_ops_total{{device="{dev}",op="read"}} {rd_ios}',
                f'host_disk_io_ops_total{{device="{dev}",op="write"}} {wr_ios}',
                "# HELP host_disk_device_info Primary /home block device name.",
                "# TYPE host_disk_device_info gauge",
                f'host_disk_device_info{{device="{dev}",mount="/home"}} 1',
            ]
        )

    return lines


def dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    key = str(path)
    now = time.monotonic()
    cached = _dir_bytes_cache.get(key)
    if cached is not None and now - cached[0] < _DIR_BYTES_TTL_SEC:
        return cached[1]
    out = run(["du", "-sb", str(path)])
    if not out:
        return 0
    nbytes = int(out.split()[0])
    _dir_bytes_cache[key] = (now, nbytes)
    return nbytes


def count_shards(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("model-*.safetensors"))


def disk_stat(path: Path) -> tuple[int, int, int]:
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, usage.free


def read_progress(work: Path) -> dict:
    p = work / "glm-merge-progress.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def parse_log_phase(work: Path) -> tuple[int, str, int]:
    """Returns (phase_id, phase_label, merge_shard_index)."""
    merge_log = work / "glm-merge.log"
    orch_log = work / "glm-merge-orchestrator.log"
    progress = read_progress(work)

    if progress.get("complete"):
        return PHASE_COMPLETE, "complete", int(progress.get("b_shard_index", SHARD_TOTAL))

    if progress.get("phase") == "merge":
        return PHASE_MERGE, "merging", int(progress.get("b_shard_index", 0))

    text = ""
    for p in (merge_log, orch_log):
        if p.exists():
            try:
                text += p.read_text(errors="replace")[-8000:]
            except OSError:
                pass

    if "starting streamed SLERP merge" in text or "B shard model-" in text:
        m = re.findall(r"B shard model-\d+-of-\d+", text)
        idx = len(m)
        return PHASE_MERGE, "merging", idx

    if proc_running("glm_stream_merge.py"):
        return PHASE_MERGE, "merging", int(progress.get("b_shard_index", 0))

    if proc_running("zai-org/GLM-5.1", "hf download zai-org/GLM-5.1") or "waiting for GLM-5.1 download" in text:
        return PHASE_DOWNLOAD, "downloading_glm51", 0

    if "merge complete" in text.lower():
        return PHASE_COMPLETE, "complete", SHARD_TOTAL

    return PHASE_IDLE, "idle", 0


def collect_metrics() -> str:
    models = env_path("GLM_MODELS_DIR", "~/models")
    work = env_path("GLM_WORK_DIR", "~/work")
    a_dir = models / "GLM-5.1"
    out_dir = models / "GLM-5.1-5.2-merged"
    b_cache = models / "GLM-5.2-cache"

    a_shards = count_shards(a_dir)
    out_shards = count_shards(out_dir)
    a_bytes = dir_bytes(a_dir)
    out_bytes = dir_bytes(out_dir)
    b_cache_bytes = dir_bytes(b_cache)
    total_disk, used_disk, free_disk = disk_stat(models if models.exists() else Path("/home"))

    phase_id, phase_label, merge_shard = parse_log_phase(work)
    progress = read_progress(work)

    download_running = 1 if proc_running("zai-org/GLM-5.1", "hf download zai-org/GLM-5.1") else 0
    merge_running = 1 if proc_running("glm_stream_merge.py") else 0
    orch_running = 1 if proc_running("run_glm_merge_remote.sh") else 0

    download_pct = min(100.0, (a_bytes / TARGET_BYTES) * 100.0)
    merge_pct = min(100.0, (merge_shard / SHARD_TOTAL) * 100.0)
    if phase_id == PHASE_COMPLETE:
        merge_pct = 100.0
        download_pct = 100.0

    lines = [
        "# HELP glm_pipeline_info Static pipeline metadata.",
        "# TYPE glm_pipeline_info gauge",
        'glm_pipeline_info{model_a="GLM-5.1",model_b="GLM-5.2",method="slerp"} 1',
        "# HELP glm_phase Pipeline phase (0=idle 1=download 2=merge 3=complete).",
        "# TYPE glm_phase gauge",
        f'glm_phase{{label="{phase_label}"}} {phase_id}',
        "# HELP glm_download_shards_complete GLM-5.1 safetensors shards on disk.",
        "# TYPE glm_download_shards_complete gauge",
        f"glm_download_shards_complete {a_shards}",
        "# HELP glm_download_shards_total Expected shard count.",
        "# TYPE glm_download_shards_total gauge",
        f"glm_download_shards_total {SHARD_TOTAL}",
        "# HELP glm_download_bytes GLM-5.1 directory size in bytes.",
        "# TYPE glm_download_bytes gauge",
        f"glm_download_bytes {a_bytes}",
        "# HELP glm_download_percent Download progress toward ~1.5TB.",
        "# TYPE glm_download_percent gauge",
        f"glm_download_percent {download_pct:.4f}",
        "# HELP glm_merge_shard_index Current B-shard index during stream merge.",
        "# TYPE glm_merge_shard_index gauge",
        f"glm_merge_shard_index {merge_shard}",
        "# HELP glm_merge_shards_total Total B shards to stream.",
        "# TYPE glm_merge_shards_total gauge",
        f"glm_merge_shards_total {SHARD_TOTAL}",
        "# HELP glm_merge_percent Stream-merge progress percent.",
        "# TYPE glm_merge_percent gauge",
        f"glm_merge_percent {merge_pct:.4f}",
        "# HELP glm_output_shards_complete Merged output shards written.",
        "# TYPE glm_output_shards_complete gauge",
        f"glm_output_shards_complete {out_shards}",
        "# HELP glm_output_bytes Merged model directory size in bytes.",
        "# TYPE glm_output_bytes gauge",
        f"glm_output_bytes {out_bytes}",
        "# HELP glm_b_cache_bytes Temporary GLM-5.2 shard cache size.",
        "# TYPE glm_b_cache_bytes gauge",
        f"glm_b_cache_bytes {b_cache_bytes}",
        "# HELP glm_tensors_merged Cumulative tensors blended (from progress file).",
        "# TYPE glm_tensors_merged counter",
        f"glm_tensors_merged {int(progress.get('tensors_merged', 0))}",
        "# HELP glm_process_running Whether a pipeline process is running.",
        "# TYPE glm_process_running gauge",
        f'glm_process_running{{process="download"}} {download_running}',
        f'glm_process_running{{process="merge"}} {merge_running}',
        f'glm_process_running{{process="orchestrator"}} {orch_running}',
        "# HELP glm_disk_bytes Disk usage on models volume.",
        "# TYPE glm_disk_bytes gauge",
        f'glm_disk_bytes{{kind="total"}} {total_disk}',
        f'glm_disk_bytes{{kind="used"}} {used_disk}',
        f'glm_disk_bytes{{kind="free"}} {free_disk}',
        "# HELP glm_disk_headroom_gib Free disk vs peak merge need (~3.0 TiB).",
        "# TYPE glm_disk_headroom_gib gauge",
        f"glm_disk_headroom_gib {(free_disk / (1024**3)) - 3072:.2f}",
        "# HELP glm_exporter_up Exporter scrape health.",
        "# TYPE glm_exporter_up gauge",
        "glm_exporter_up 1",
        "# HELP glm_exporter_scrape_timestamp_seconds Unix time of this scrape.",
        "# TYPE glm_exporter_scrape_timestamp_seconds gauge",
        f"glm_exporter_scrape_timestamp_seconds {time.time():.3f}",
    ]
    lines.extend(host_metrics_lines())
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        body = collect_metrics().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def main() -> None:
    port = int(os.environ.get("GLM_EXPORTER_PORT", "9108"))
    host = os.environ.get("GLM_EXPORTER_HOST", "0.0.0.0")
    server = HTTPServer((host, port), Handler)
    print(f"glm merge exporter on http://{host}:{port}/metrics", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
