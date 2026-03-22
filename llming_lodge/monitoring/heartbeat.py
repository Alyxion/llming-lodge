"""Event loop health monitor.

Measures asyncio event loop responsiveness by scheduling frequent sleeps
and recording actual elapsed time. Also tracks CPU, memory, disk, and
process-level metrics for the dev dashboard.
"""

import asyncio
import logging
import os
import platform
import shutil
import sys
import time
import uuid
from collections import deque
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)

_TICK_INTERVAL = 0.05  # 50ms target sleep
_JITTER_THRESHOLD_MS = 2.0  # only record delays above this
_RAW_MAXLEN = 1000  # ~10s of raw samples
_AGG_MAXLEN = 600  # 10 minutes of 1-second buckets
_SYS_MAXLEN = 600
_LOG_INTERVAL = 30  # seconds between summary log lines


class HeartbeatMonitor:
    _instance: "HeartbeatMonitor | None" = None

    @classmethod
    def get(cls) -> "HeartbeatMonitor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._start_time: float = 0.0
        self._raw: deque[tuple[float, float]] = deque(maxlen=_RAW_MAXLEN)
        self._heartbeat: deque[dict] = deque(maxlen=_AGG_MAXLEN)
        self._system: deque[dict] = deque(maxlen=_SYS_MAXLEN)
        self._last_agg_time: float = 0.0
        self._last_log_time: float = 0.0
        self._server_name = platform.node() or "unknown"
        self.worker_id = str(uuid.uuid4())
        self._proc = psutil.Process()
        self._info_cache: dict | None = None
        self._info_cache_time: float = 0.0

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._start_time = time.monotonic()
        self._last_agg_time = time.monotonic()
        # Prime psutil's cpu_percent (first call always returns 0)
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent()
        # Pre-fill buffers so graphs show a full 10-min window immediately
        self._prefill()
        self._task = asyncio.create_task(self._loop())
        logger.info("[HEARTBEAT] Monitor started")

    def _prefill(self) -> None:
        """Fill both deques with baseline values spanning the past 10 minutes."""
        now = time.time()
        mem = psutil.virtual_memory()
        try:
            mi = self._proc.memory_info()
            proc_rss = round(mi.rss / (1024 * 1024), 1)
            proc_vms = round(mi.vms / (1024 * 1024), 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_rss = proc_vms = 0.0
        for i in range(_AGG_MAXLEN):
            ts = now - (_AGG_MAXLEN - 1 - i)
            self._heartbeat.append({
                "ts": ts, "avg_ms": 0.0, "max_ms": 0.0, "count": 0, "p95_ms": 0.0,
            })
            self._system.append({
                "ts": ts,
                "cpu_percent": 0.0,
                "per_cpu": [],
                "load_avg": None,
                "mem_used_mb": round(mem.used / (1024 * 1024), 1),
                "mem_total_mb": round(mem.total / (1024 * 1024), 1),
                "mem_percent": mem.percent,
                "proc_rss_mb": proc_rss,
                "proc_vms_mb": proc_vms,
                "proc_cpu": 0.0,
            })

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("[HEARTBEAT] Monitor stopped")

    async def _loop(self) -> None:
        self._last_log_time = time.monotonic()
        try:
            while True:
                t0 = time.monotonic()
                await asyncio.sleep(_TICK_INTERVAL)
                delay_ms = (time.monotonic() - t0) * 1000 - _TICK_INTERVAL * 1000
                if delay_ms > _JITTER_THRESHOLD_MS:
                    self._raw.append((time.time(), delay_ms))

                now = time.monotonic()
                if now - self._last_agg_time >= 1.0:
                    self._aggregate(now)
                    self._sample_system()
                    self._last_agg_time = now

                if now - self._last_log_time >= _LOG_INTERVAL:
                    self._log_summary()
                    self._last_log_time = now
        except asyncio.CancelledError:
            pass

    def _log_summary(self) -> None:
        """Log a compact health summary over the last 30s window."""
        cutoff = time.time() - _LOG_INTERVAL

        # Event loop delay stats from heartbeat buckets
        recent_hb = [b for b in self._heartbeat if b["ts"] > cutoff and b["count"] > 0]
        if recent_hb:
            all_avg = [b["avg_ms"] for b in recent_hb]
            all_max = [b["max_ms"] for b in recent_hb]
            loop_avg = round(sum(all_avg) / len(all_avg), 1)
            loop_min = round(min(all_avg), 1)
            loop_max = round(max(all_max), 1)
            jitter_total = sum(b["count"] for b in recent_hb)
            loop_str = f"avg={loop_avg} min={loop_min} max={loop_max} jitters={jitter_total}"
        else:
            loop_str = "avg=0 min=0 max=0 jitters=0"

        # System metrics from recent window
        recent_sys = [s for s in self._system if s["ts"] > cutoff]
        if recent_sys:
            cpu_vals = [s["cpu_percent"] for s in recent_sys]
            cpu_avg = round(sum(cpu_vals) / len(cpu_vals), 1)
            cpu_max = round(max(cpu_vals), 1)
            mem_cur = recent_sys[-1]["mem_percent"]
            mem_max = round(max(s["mem_percent"] for s in recent_sys), 1)
            mem_used = recent_sys[-1]["mem_used_mb"]
            mem_total = recent_sys[-1]["mem_total_mb"]
            proc_rss = recent_sys[-1]["proc_rss_mb"]
            proc_rss_max = round(max(s["proc_rss_mb"] for s in recent_sys), 1)
        else:
            cpu_avg = cpu_max = mem_cur = mem_max = mem_used = mem_total = 0
            proc_rss = proc_rss_max = 0

        # Per-CPU load (snapshot)
        cpu_count = psutil.cpu_count(logical=True) or 0
        try:
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            per_cpu_str = ",".join(f"{v:.0f}" for v in per_cpu)
        except Exception:
            per_cpu_str = "?"

        # Load average
        try:
            la = os.getloadavg()
            load_str = f" | load={la[0]:.1f}/{la[1]:.1f}/{la[2]:.1f}"
        except (OSError, AttributeError):
            load_str = ""

        # Disk usage — root partition only
        disk_str = ""
        try:
            du = shutil.disk_usage("/")
            disk_pct = round(du.used / du.total * 100, 1) if du.total else 0
            disk_free_gb = round(du.free / (1024**3), 1)
            disk_str = f" | disk=/{disk_pct}%({disk_free_gb}GB free)"
        except OSError:
            pass

        uptime = round(time.monotonic() - self._start_time) if self._start_time else 0

        logger.info(
            f"[HEALTH] loop[{loop_str}]ms"
            f" | cpu={cpu_avg}/{cpu_max}% x{cpu_count}[{per_cpu_str}]"
            f"{load_str}"
            f" | mem={mem_cur}/{mem_max}%({mem_used:.0f}/{mem_total:.0f}MB)"
            f" | proc={proc_rss}/{proc_rss_max}MB"
            f"{disk_str}"
            f" | up={uptime}s"
        )

    def _aggregate(self, now: float) -> None:
        cutoff = time.time() - 1.0
        samples = [ms for ts, ms in self._raw if ts >= cutoff]
        if not samples:
            bucket = {
                "ts": time.time(),
                "avg_ms": 0.0,
                "max_ms": 0.0,
                "count": 0,
                "p95_ms": 0.0,
            }
        else:
            samples_sorted = sorted(samples)
            p95_idx = max(0, int(len(samples_sorted) * 0.95) - 1)
            bucket = {
                "ts": time.time(),
                "avg_ms": round(sum(samples) / len(samples), 2),
                "max_ms": round(max(samples), 2),
                "count": len(samples),
                "p95_ms": round(samples_sorted[p95_idx], 2),
            }
        self._heartbeat.append(bucket)

    def _sample_system(self) -> None:
        mem = psutil.virtual_memory()
        try:
            mi = self._proc.memory_info()
            proc_rss = round(mi.rss / (1024 * 1024), 1)
            proc_vms = round(mi.vms / (1024 * 1024), 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_rss = proc_vms = 0.0
        try:
            proc_cpu = self._proc.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_cpu = 0.0
        try:
            per_cpu = [round(v, 1) for v in psutil.cpu_percent(interval=None, percpu=True)]
        except Exception:
            per_cpu = []
        try:
            la = os.getloadavg()
            load_avg = [round(x, 2) for x in la]
        except (OSError, AttributeError):
            load_avg = None
        self._system.append({
            "ts": time.time(),
            "cpu_percent": sum(per_cpu) / len(per_cpu) if per_cpu else 0.0,
            "per_cpu": per_cpu,
            "load_avg": load_avg,
            "mem_used_mb": round(mem.used / (1024 * 1024), 1),
            "mem_total_mb": round(mem.total / (1024 * 1024), 1),
            "mem_percent": mem.percent,
            "proc_rss_mb": proc_rss,
            "proc_vms_mb": proc_vms,
            "proc_cpu": proc_cpu,
        })

    def get_state(self, since: float = 0) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        hb = [b for b in self._heartbeat if b["ts"] > since]
        sys_data = [s for s in self._system if s["ts"] > since]
        return {
            "server_name": self._server_name,
            "worker_id": self.worker_id,
            "uptime_s": round(uptime, 1),
            "heartbeat": hb,
            "system": sys_data,
            "now": time.time(),
        }

    def snapshot_asyncio_tasks(self) -> list[dict]:
        """Snapshot asyncio tasks — must be called from the event loop thread."""
        tasks = []
        try:
            for t in asyncio.all_tasks():
                coro = t.get_coro()
                tasks.append({
                    "name": t.get_name(),
                    "coro": getattr(coro, "__qualname__", str(coro)),
                    "done": t.done(),
                })
        except RuntimeError:
            pass
        return tasks

    def get_info(self, asyncio_tasks: list[dict] | None = None) -> dict:
        """Collect detailed system, process, disk, and asyncio info (semi-static).

        asyncio_tasks should be pre-captured from the event loop thread via
        snapshot_asyncio_tasks() since this method may run in a threadpool.
        Results are cached for 10 seconds (except asyncio tasks which are always fresh).
        """
        now = time.monotonic()
        if self._info_cache and now - self._info_cache_time < 10.0:
            # Return cached result with fresh asyncio tasks
            cached = self._info_cache.copy()
            cached["asyncio_tasks"] = asyncio_tasks or []
            cached["asyncio_task_count"] = len(cached["asyncio_tasks"])
            return cached

        proc = self._proc

        # ── Process info ──
        try:
            proc_info = {
                "pid": proc.pid,
                "ppid": proc.ppid(),
                "name": proc.name(),
                "cmdline": " ".join(proc.cmdline()[:6]),
                "cwd": proc.cwd(),
                "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
                "num_threads": proc.num_threads(),
                "status": proc.status(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_info = {"pid": os.getpid(), "error": "access denied"}

        try:
            proc_info["num_fds"] = proc.num_fds()
        except (AttributeError, psutil.NoSuchProcess, psutil.AccessDenied):
            proc_info["num_fds"] = -1

        try:
            conns = proc.net_connections(kind="inet")
            conn_summary = {}
            for c in conns:
                st = c.status
                conn_summary[st] = conn_summary.get(st, 0) + 1
            proc_info["connections"] = conn_summary
            proc_info["connection_count"] = len(conns)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            proc_info["connections"] = {}
            proc_info["connection_count"] = -1

        try:
            children = proc.children(recursive=True)
            proc_info["children"] = [
                {"pid": c.pid, "name": c.name(), "status": c.status()}
                for c in children
            ]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_info["children"] = []

        mi = proc.memory_info()
        proc_info["mem_rss_mb"] = round(mi.rss / (1024 * 1024), 1)
        proc_info["mem_vms_mb"] = round(mi.vms / (1024 * 1024), 1)

        try:
            ct = proc.cpu_times()
            proc_info["cpu_user_s"] = round(ct.user, 2)
            proc_info["cpu_system_s"] = round(ct.system, 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # ── Python / runtime ──
        runtime_info = {
            "python_version": sys.version.split()[0],
            "python_impl": platform.python_implementation(),
            "platform": platform.platform(),
            "arch": platform.machine(),
            "hostname": platform.node(),
        }

        # ── System info ──
        vm = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        sys_info = {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq_mhz": round(cpu_freq.current, 0) if cpu_freq else None,
            "mem_total_mb": round(vm.total / (1024 * 1024), 0),
            "mem_available_mb": round(vm.available / (1024 * 1024), 0),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        }

        try:
            la = os.getloadavg()
            sys_info["load_avg"] = [round(x, 2) for x in la]
        except (OSError, AttributeError):
            sys_info["load_avg"] = None

        # ── Disk ──
        disks = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = shutil.disk_usage(part.mountpoint)
                disks.append({
                    "mount": part.mountpoint,
                    "device": part.device,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / (1024**3), 1),
                    "used_gb": round(usage.used / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                    "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0,
                })
            except (PermissionError, OSError):
                pass
        sys_info["disks"] = disks

        # ── Network interfaces ──
        net_ifs = []
        addrs = psutil.net_if_addrs()
        for name, entries in addrs.items():
            ips = []
            for e in entries:
                if e.family.name in ("AF_INET", "AF_INET6"):
                    ips.append({"family": e.family.name, "addr": e.address})
            if ips:
                net_ifs.append({"name": name, "addrs": ips})
        sys_info["network_interfaces"] = net_ifs

        # ── Asyncio tasks (pre-captured from event loop thread) ──
        loop_tasks = asyncio_tasks or []

        # ── Environment (filtered — only safe keys) ──
        safe_prefixes = (
            "NICEGUI_", "DEBUG_", "LLMING_", "WEBSITE_",
            "PORT", "PYTHON", "PATH", "HOME", "USER", "LANG",
        )
        env_filtered = {}
        for k, v in sorted(os.environ.items()):
            if any(k.startswith(p) for p in safe_prefixes):
                # Mask values that look like secrets
                if any(s in k.upper() for s in ("PASSWORD", "SECRET", "KEY", "TOKEN")):
                    env_filtered[k] = v[:2] + "***"
                else:
                    env_filtered[k] = v

        result = {
            "process": proc_info,
            "runtime": runtime_info,
            "system": sys_info,
            "asyncio_tasks": loop_tasks,
            "asyncio_task_count": len(loop_tasks),
            "environment": env_filtered,
        }
        self._info_cache = result
        self._info_cache_time = time.monotonic()
        return result
