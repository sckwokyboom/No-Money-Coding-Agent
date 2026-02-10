#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class Progress:
    def __init__(self, enabled: bool = True, min_interval: float = 0.2) -> None:
        self.enabled = enabled
        self.min_interval = min_interval
        self.started_at = time.time()
        self.last_update = 0.0
        self.last_len = 0

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        sec = max(0, int(seconds))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(self, stage: str, done: int, total: int, extra: str = "", force: bool = False) -> None:
        if not self.enabled:
            return

        now = time.time()
        if not force and (now - self.last_update) < self.min_interval:
            return
        self.last_update = now

        elapsed = max(1e-9, now - self.started_at)
        rate = done / elapsed

        pct = (100.0 * done / total) if total > 0 else 0.0
        line = f"[{stage}] {done}/{total} ({pct:5.1f}%)"

        if rate > 0:
            line += f" | {rate:.1f} rows/s"
        if total > 0 and done <= total and rate > 0:
            eta = (total - done) / rate
            line += f" | eta {self._fmt_seconds(eta)}"
        if extra:
            line += f" | {extra}"

        out = line[:500]
        pad = max(0, self.last_len - len(out))
        sys.stderr.write("\r" + out + (" " * pad))
        sys.stderr.flush()
        self.last_len = len(out)

    def done(self) -> None:
        if not self.enabled:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.last_len = 0


def load_yaml(path: str) -> Dict[str, Any]:
    if not yaml:
        raise RuntimeError("PyYAML is not installed. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def count_nonempty_lines(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def parse_owner_repo(repo: Optional[str]) -> Optional[Tuple[str, str]]:
    if not repo or "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    owner = owner.strip()
    name = name.strip()
    if not owner or not name:
        return None
    return owner, name


def load_token(cli_token: str, token_file: str) -> Optional[str]:
    if cli_token.strip():
        return cli_token.strip()
    if token_file.strip():
        with open(token_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    env = os.getenv("GITHUB_TOKEN")
    return env.strip() if env else None


def gh_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "nmc-agent-spring-stats",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _truncate(s: str, max_len: int = 32) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


@dataclass
class GhRequestResult:
    data: Optional[Dict[str, Any]]
    status_code: int
    latency_ms: float
    error_kind: str
    rate_remaining: Optional[int]
    rate_reset_epoch: Optional[int]
    detail: str = ""


class GitHubStageReporter:
    def __init__(self, *, enabled: bool, total: int, heartbeat_seconds: float) -> None:
        self.enabled = enabled
        self.total = total
        self.heartbeat_seconds = max(0.2, heartbeat_seconds)
        self.progress = Progress(enabled=enabled)

        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.done = 0
        self.current_repo = "-"
        self.current_endpoint = "-"
        self.current_request_started_at: Optional[float] = None
        self.last_success_at = time.time()
        self.retry_attempt = 0
        self.retry_total = 0
        self.retry_reason = ""
        self.sleep_until: Optional[float] = None
        self.sleep_reason = ""
        self.rate_remaining: Optional[int] = None
        self.rate_reset_epoch: Optional[int] = None
        self.counts = Counter({"ok": 0, "404": 0, "403": 0, "timeouts": 0, "5xx": 0, "other": 0})

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._heartbeat_loop, name="gh-progress-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.emit(force=True)
        self.progress.done()

    def _heartbeat_loop(self) -> None:
        next_emit = 0.0
        while not self._stop_evt.wait(0.2):
            now = time.time()
            if now >= next_emit:
                self.emit()
                next_emit = now + self.heartbeat_seconds

    def set_current_request(self, *, repo: str, endpoint: str) -> None:
        with self._lock:
            self.current_repo = repo
            self.current_endpoint = endpoint
            self.current_request_started_at = time.time()

    def finish_current_request(self, *, ok: bool) -> None:
        with self._lock:
            self.current_request_started_at = None
            if ok:
                self.last_success_at = time.time()

    def mark_repo_done(self, done: int) -> None:
        with self._lock:
            self.done = done
            self.current_endpoint = "-"

    def set_retry(self, *, attempt: int, total: int, reason: str) -> None:
        with self._lock:
            self.retry_attempt = attempt
            self.retry_total = total
            self.retry_reason = reason

    def clear_retry(self) -> None:
        with self._lock:
            self.retry_attempt = 0
            self.retry_total = 0
            self.retry_reason = ""

    def set_sleep(self, *, reason: str, seconds: float) -> None:
        with self._lock:
            self.sleep_reason = reason
            self.sleep_until = time.time() + max(0.0, seconds)

    def clear_sleep(self) -> None:
        with self._lock:
            self.sleep_reason = ""
            self.sleep_until = None

    def bump_count(self, key: str) -> None:
        with self._lock:
            if key in self.counts:
                self.counts[key] += 1
            else:
                self.counts["other"] += 1

    def set_rate_limit(self, remaining: Optional[int], reset_epoch: Optional[int]) -> None:
        with self._lock:
            if remaining is not None:
                self.rate_remaining = remaining
            if reset_epoch is not None:
                self.rate_reset_epoch = reset_epoch

    def log_line(self, message: str) -> None:
        with self._io_lock:
            if self.enabled:
                sys.stderr.write("\n")
            sys.stderr.write(message.rstrip() + "\n")
            sys.stderr.flush()

    def emit(self, force: bool = False) -> None:
        if not self.enabled:
            return

        with self._lock:
            now = time.time()
            done = self.done
            total = self.total
            repo = self.current_repo
            endpoint = self.current_endpoint
            req_started = self.current_request_started_at
            last_success_at = self.last_success_at
            retry_attempt = self.retry_attempt
            retry_total = self.retry_total
            retry_reason = self.retry_reason
            sleep_until = self.sleep_until
            sleep_reason = self.sleep_reason
            rate_remaining = self.rate_remaining
            counts = dict(self.counts)

        parts: List[str] = [
            f"repo={_truncate(repo, 42)}",
            f"ep={endpoint}",
            f"ok={counts.get('ok', 0)}",
            f"404={counts.get('404', 0)}",
            f"403={counts.get('403', 0)}",
            f"timeouts={counts.get('timeouts', 0)}",
            f"5xx={counts.get('5xx', 0)}",
            f"other={counts.get('other', 0)}",
        ]

        if rate_remaining is not None:
            parts.append(f"rl={rate_remaining}")

        if req_started is not None:
            parts.append(f"req_age={max(0.0, now - req_started):.1f}s")
        else:
            parts.append(f"last_ok_age={max(0.0, now - last_success_at):.1f}s")

        if retry_attempt > 0:
            parts.append(f"retry={retry_attempt}/{retry_total}")
            parts.append(f"retry_reason={retry_reason}")

        if sleep_until is not None and sleep_until > now:
            parts.append(f"sleep={max(0.0, sleep_until - now):.1f}s")
            parts.append(f"sleep_reason={sleep_reason}")

        extra = " ".join(parts)
        with self._io_lock:
            self.progress.update("stats-github", done, total, extra=extra, force=force)


def gh_get_json_once(
    url: str,
    headers: Dict[str, str],
    *,
    connect_timeout: float,
    read_timeout: float,
    ca_bundle: str,
    insecure: bool,
) -> GhRequestResult:
    verify: Any = True
    if insecure:
        verify = False
    elif ca_bundle:
        verify = ca_bundle

    started = time.perf_counter()
    try:
        r = requests.get(url, headers=headers, timeout=(connect_timeout, read_timeout), verify=verify)
    except requests.exceptions.Timeout as exc:
        return GhRequestResult(
            data=None,
            status_code=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_kind="timeout",
            rate_remaining=None,
            rate_reset_epoch=None,
            detail=str(exc),
        )
    except requests.exceptions.SSLError as exc:
        return GhRequestResult(
            data=None,
            status_code=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_kind="ssl_error",
            rate_remaining=None,
            rate_reset_epoch=None,
            detail=str(exc),
        )
    except requests.exceptions.ConnectionError as exc:
        return GhRequestResult(
            data=None,
            status_code=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_kind="connect_error",
            rate_remaining=None,
            rate_reset_epoch=None,
            detail=str(exc),
        )
    except requests.exceptions.RequestException as exc:
        return GhRequestResult(
            data=None,
            status_code=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_kind="unknown",
            rate_remaining=None,
            rate_reset_epoch=None,
            detail=str(exc),
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    remaining = _safe_int(r.headers.get("X-RateLimit-Remaining"))
    reset_epoch = _safe_int(r.headers.get("X-RateLimit-Reset"))

    if r.status_code == 403:
        txt = (r.text or "").lower()
        if "abuse" in txt or "secondary rate limit" in txt:
            return GhRequestResult(None, r.status_code, latency_ms, "abuse_detected", remaining, reset_epoch, r.reason)
        if remaining == 0 or "rate limit" in txt:
            return GhRequestResult(None, r.status_code, latency_ms, "rate_limited", remaining, reset_epoch, r.reason)
        return GhRequestResult(None, r.status_code, latency_ms, "http_4xx", remaining, reset_epoch, r.reason)

    if r.status_code >= 500:
        return GhRequestResult(None, r.status_code, latency_ms, "http_5xx", remaining, reset_epoch, r.reason)

    if r.status_code >= 400:
        return GhRequestResult(None, r.status_code, latency_ms, "http_4xx", remaining, reset_epoch, r.reason)

    try:
        data = r.json()
    except Exception as exc:
        return GhRequestResult(None, r.status_code, latency_ms, "json_error", remaining, reset_epoch, str(exc))

    if not isinstance(data, dict):
        return GhRequestResult(None, r.status_code, latency_ms, "json_error", remaining, reset_epoch, "JSON payload is not an object")

    return GhRequestResult(data, r.status_code, latency_ms, "ok", remaining, reset_epoch, r.reason)


def calc_backoff_seconds(attempt: int, *, cap: float) -> float:
    exp = min(cap, 2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, 0.5)
    return min(cap, exp + jitter)


def classify_progress_bucket(result: GhRequestResult) -> str:
    if result.error_kind == "ok":
        return "ok"
    if result.status_code == 404:
        return "404"
    if result.status_code == 403:
        return "403"
    if result.error_kind == "timeout":
        return "timeouts"
    if result.error_kind == "http_5xx":
        return "5xx"
    return "other"


def summarize_result_category(result: GhRequestResult) -> List[str]:
    cats: List[str] = []
    if result.error_kind == "ok":
        cats.append("ok")
    if result.status_code == 404:
        cats.append("404")
    if result.status_code == 403:
        cats.append("403")
        if result.error_kind in ("rate_limited", "abuse_detected"):
            cats.append("rate_limited")
        else:
            cats.append("forbidden_not_authorized")
    if result.error_kind == "timeout":
        cats.append("timeouts")
    if result.error_kind == "ssl_error":
        cats.append("ssl_errors")
    if result.error_kind == "connect_error":
        cats.append("connect_errors")
    if result.error_kind == "http_5xx":
        cats.append("5xx")
    if result.error_kind == "http_4xx" and result.status_code not in (403, 404):
        cats.append("other_4xx")
    if result.error_kind == "json_error":
        cats.append("json_errors")
    if not cats:
        cats.append("other")
    return cats


def add_example(error_examples: Dict[str, List[str]], category: str, repo: str, max_examples: int) -> None:
    arr = error_examples.setdefault(category, [])
    if len(arr) >= max_examples:
        return
    if repo in arr:
        return
    arr.append(repo)


def sleep_with_observability(seconds: float, reporter: GitHubStageReporter) -> None:
    end = time.time() + max(0.0, seconds)
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))


def gh_get_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    repo: str,
    endpoint: str,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    ca_bundle: str,
    insecure: bool,
    debug_http: bool,
    reporter: GitHubStageReporter,
) -> GhRequestResult:
    attempts = max(1, max_retries)
    last = GhRequestResult(
        data=None,
        status_code=0,
        latency_ms=0.0,
        error_kind="unknown",
        rate_remaining=None,
        rate_reset_epoch=None,
    )

    for attempt in range(1, attempts + 1):
        reporter.set_current_request(repo=repo, endpoint=endpoint)
        result = gh_get_json_once(
            url,
            headers,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            ca_bundle=ca_bundle,
            insecure=insecure,
        )
        reporter.finish_current_request(ok=(result.error_kind == "ok"))
        reporter.set_rate_limit(result.rate_remaining, result.rate_reset_epoch)
        last = result

        if debug_http:
            reporter.log_line(
                f"[debug-http] repo={repo} ep={endpoint} url={url} attempt={attempt}/{attempts} "
                f"status={result.status_code} kind={result.error_kind} latency_ms={result.latency_ms:.1f} "
                f"rl_remaining={result.rate_remaining} rl_reset={result.rate_reset_epoch}"
            )

        retry_reason = ""
        sleep_seconds = 0.0

        if result.error_kind in ("timeout", "connect_error", "ssl_error", "http_5xx", "unknown"):
            retry_reason = result.error_kind
            sleep_seconds = calc_backoff_seconds(attempt, cap=30.0)
        elif result.error_kind in ("rate_limited", "abuse_detected"):
            retry_reason = result.error_kind
            if result.error_kind == "rate_limited" and result.rate_reset_epoch is not None:
                now_epoch = int(time.time())
                sleep_seconds = float(max(1, result.rate_reset_epoch - now_epoch + 1))
            else:
                sleep_seconds = calc_backoff_seconds(attempt, cap=90.0)

        if retry_reason and attempt < attempts:
            reporter.set_retry(attempt=attempt, total=attempts, reason=retry_reason)
            reporter.set_sleep(reason=retry_reason, seconds=sleep_seconds)
            if reporter.enabled or retry_reason in ("rate_limited", "abuse_detected") or sleep_seconds >= 5.0:
                if retry_reason == "rate_limited":
                    if result.rate_reset_epoch is not None:
                        reporter.log_line(
                            f"[backoff] repo={repo} ep={endpoint} retry={attempt}/{attempts} reason=rate_limited "
                            f"remaining={result.rate_remaining} sleeping={sleep_seconds:.1f}s until reset"
                        )
                    else:
                        reporter.log_line(
                            f"[backoff] repo={repo} ep={endpoint} retry={attempt}/{attempts} reason=rate_limited "
                            f"remaining={result.rate_remaining} sleeping={sleep_seconds:.1f}s"
                        )
                else:
                    reporter.log_line(
                        f"[backoff] repo={repo} ep={endpoint} retry={attempt}/{attempts} reason={retry_reason} "
                        f"sleeping={sleep_seconds:.1f}s"
                    )
            sleep_with_observability(sleep_seconds, reporter)
            reporter.clear_sleep()
            reporter.clear_retry()
            continue

        reporter.clear_sleep()
        reporter.clear_retry()
        return result

    return last


@dataclass
class RepoAgg:
    samples: int = 0
    max_final_score: int = 0
    max_patch_score: int = 0


def bucket_java_share(x: float) -> str:
    if x >= 0.75:
        return ">=75%"
    if x >= 0.50:
        return "50-75%"
    if x >= 0.25:
        return "25-50%"
    if x >= 0.10:
        return "10-25%"
    if x > 0:
        return "0-10%"
    return "0%"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute stats over spring_only.jsonl (+ optional GitHub checks).")
    ap.add_argument("--config", default="", help="YAML config path")

    ap.add_argument("--input", required=False, default="", help="spring_only.jsonl path")
    ap.add_argument("--top", type=int, default=20)

    ap.add_argument("--github", action="store_true")
    ap.add_argument("--repo-info", action="store_true")
    ap.add_argument("--max-repos", type=int, default=0)

    ap.add_argument("--token", default="")
    ap.add_argument("--token-file", default="")

    ap.add_argument("--ca-bundle", default="")
    ap.add_argument("--insecure", action="store_true")

    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=30.0, help="Read timeout seconds (backward-compatible alias).")
    ap.add_argument("--read-timeout", type=float, default=None, help="Read timeout seconds.")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--debug-http", action="store_true")
    ap.add_argument("--error-examples", type=int, default=5)

    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument("--heartbeat-seconds", type=float, default=2.0)

    return ap


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    if not args.input:
        args.input = str(deep_get(cfg, ["input"], "") or "")
    args.top = int(deep_get(cfg, ["top"], args.top))

    gh_enabled = deep_get(cfg, ["github", "enabled"], None)
    if gh_enabled is not None and not args.github:
        args.github = bool(gh_enabled)
    args.repo_info = bool(deep_get(cfg, ["github", "repo_info"], args.repo_info))
    args.max_repos = int(deep_get(cfg, ["github", "max_repos"], args.max_repos))

    if not args.token:
        args.token = str(deep_get(cfg, ["github", "token_source", "token"], "") or "")
    if not args.token_file:
        args.token_file = str(deep_get(cfg, ["github", "token_source", "token_file"], "") or "")

    if not args.ca_bundle:
        args.ca_bundle = str(deep_get(cfg, ["github", "tls", "ca_bundle"], "") or "")
    args.insecure = bool(deep_get(cfg, ["github", "tls", "insecure"], args.insecure))
    args.debug_http = bool(deep_get(cfg, ["github", "debug_http"], args.debug_http))
    args.error_examples = int(deep_get(cfg, ["github", "error_examples"], args.error_examples))

    args.connect_timeout = float(deep_get(cfg, ["http", "connect_timeout"], args.connect_timeout))
    cfg_read_timeout = deep_get(cfg, ["http", "read_timeout"], None)
    cfg_timeout_alias = deep_get(cfg, ["http", "timeout"], None)
    if cfg_read_timeout is not None:
        args.read_timeout = float(cfg_read_timeout)
    elif cfg_timeout_alias is not None:
        args.read_timeout = float(cfg_timeout_alias)
    elif args.read_timeout is None:
        args.read_timeout = float(args.timeout)
    args.timeout = float(args.timeout)
    args.retries = int(deep_get(cfg, ["http", "retries"], args.retries))

    progress_enabled = deep_get(cfg, ["progress", "enabled"], None)
    if progress_enabled is not None and not args.progress:
        args.progress = bool(progress_enabled)
    args.progress_every = int(deep_get(cfg, ["progress", "every"], args.progress_every))
    args.heartbeat_seconds = float(deep_get(cfg, ["progress", "heartbeat_seconds"], args.heartbeat_seconds))

    return args


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    args = apply_config(args)
    if args.read_timeout is None:
        args.read_timeout = float(args.timeout)

    if not args.input:
        raise SystemExit("You must provide --input (or set it in --config).")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {args.input}")

    total_lines = count_nonempty_lines(args.input)
    progress = Progress(enabled=args.progress)

    total_samples = 0
    invalid_json = 0
    invalid_repo = 0
    repo_aggs: Dict[str, RepoAgg] = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_samples += 1
            try:
                row = json.loads(line)
            except Exception:
                invalid_json += 1
                if args.progress and (total_samples % args.progress_every == 0):
                    progress.update(
                        "stats-read",
                        total_samples,
                        total_lines,
                        extra=f"repos={len(repo_aggs)} invalid_json={invalid_json} invalid_repo={invalid_repo}",
                    )
                continue

            repo = row.get("repo")
            if not repo or "/" not in str(repo):
                invalid_repo += 1
                if args.progress and (total_samples % args.progress_every == 0):
                    progress.update(
                        "stats-read",
                        total_samples,
                        total_lines,
                        extra=f"repos={len(repo_aggs)} invalid_json={invalid_json} invalid_repo={invalid_repo}",
                    )
                continue

            repo = str(repo)
            agg = repo_aggs.get(repo)
            if agg is None:
                agg = RepoAgg()
                repo_aggs[repo] = agg
            agg.samples += 1
            agg.max_final_score = max(agg.max_final_score, int(row.get("final_score", 0) or 0))
            agg.max_patch_score = max(agg.max_patch_score, int(row.get("patch_score", 0) or 0))

            if args.progress and (total_samples % args.progress_every == 0 or total_samples == total_lines):
                progress.update(
                    "stats-read",
                    total_samples,
                    total_lines,
                    extra=f"repos={len(repo_aggs)} invalid_json={invalid_json} invalid_repo={invalid_repo}",
                )

    if args.progress:
        progress.update(
            "stats-read",
            min(total_samples, total_lines),
            total_lines,
            extra=f"repos={len(repo_aggs)} invalid_json={invalid_json} invalid_repo={invalid_repo}",
            force=True,
        )
        progress.done()

    unique_repos = len(repo_aggs)

    print("=== Local file stats ===")
    print(f"Input:         {args.input}")
    print(f"Total samples: {total_samples}")
    print(f"Unique repos:  {unique_repos}")
    print(f"Invalid JSON:  {invalid_json}")
    print(f"Invalid repo:  {invalid_repo}")
    print()

    top_repos = sorted(repo_aggs.items(), key=lambda kv: kv[1].samples, reverse=True)[: args.top]
    print(f"Top-{args.top} repos by #samples:")
    for repo, agg in top_repos:
        print(f"  {repo:40s} samples={agg.samples:5d}  max_final={agg.max_final_score:3d}  max_patch={agg.max_patch_score:3d}")
    print()

    if not args.github:
        print("GitHub checks disabled. Add --github (and optionally token/CA settings).")
        return

    token = load_token(args.token, args.token_file)
    headers = gh_headers(token)

    repos_list = list(repo_aggs.keys())
    if args.max_repos and args.max_repos > 0:
        repos_list = repos_list[: args.max_repos]

    lang_bytes_total = Counter()
    primary_lang_counter = Counter()
    java_share_bucket_gh = Counter()

    stars_total = 0
    forks_total = 0
    archived_count = 0

    ok = 0
    not_found = 0
    forbidden = 0
    other_err = Counter()
    error_counts = Counter()
    error_examples: Dict[str, List[str]] = {}
    last_rate_remaining: Optional[int] = None
    last_rate_reset_epoch: Optional[int] = None

    gh_reporter = GitHubStageReporter(
        enabled=args.progress,
        total=len(repos_list),
        heartbeat_seconds=args.heartbeat_seconds,
    )
    gh_reporter.log_line(
        f"[github-http] connect_timeout={args.connect_timeout:.1f}s read_timeout={args.read_timeout:.1f}s "
        f"retries={args.retries} heartbeat={args.heartbeat_seconds:.1f}s debug_http={args.debug_http}"
    )
    gh_reporter.start()
    try:
        for i, repo in enumerate(repos_list, 1):
            pr = parse_owner_repo(repo)
            if pr is None:
                gh_reporter.mark_repo_done(i)
                continue
            owner, name = pr

            languages_url = f"https://api.github.com/repos/{owner}/{name}/languages"
            lang_result = gh_get_with_retries(
                url=languages_url,
                headers=headers,
                repo=repo,
                endpoint="languages",
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
                max_retries=args.retries,
                ca_bundle=args.ca_bundle,
                insecure=args.insecure,
                debug_http=args.debug_http,
                reporter=gh_reporter,
            )

            if lang_result.rate_remaining is not None:
                last_rate_remaining = lang_result.rate_remaining
            if lang_result.rate_reset_epoch is not None:
                last_rate_reset_epoch = lang_result.rate_reset_epoch

            gh_reporter.bump_count(classify_progress_bucket(lang_result))
            for cat in summarize_result_category(lang_result):
                error_counts[cat] += 1
                if cat != "ok":
                    add_example(error_examples, cat, repo, args.error_examples)

            langs = lang_result.data
            status = lang_result.status_code
            if status == 404:
                not_found += 1
            elif status == 403:
                forbidden += 1
            elif not langs or status >= 400 or lang_result.error_kind != "ok":
                other_key: Any = status if status > 0 else lang_result.error_kind
                other_err[other_key] += 1
            else:
                ok += 1
                total = 0
                java_b = 0
                for k, v in langs.items():
                    try:
                        b = int(v)
                    except Exception:
                        continue
                    lang_bytes_total[k] += b
                    total += b
                    if k == "Java":
                        java_b += b

                if total > 0:
                    primary = max(langs.items(), key=lambda kv: int(kv[1]))[0]
                    primary_lang_counter[primary] += 1
                    java_share_bucket_gh[bucket_java_share(java_b / total)] += 1

            if args.repo_info:
                info_url = f"https://api.github.com/repos/{owner}/{name}"
                info_result = gh_get_with_retries(
                    url=info_url,
                    headers=headers,
                    repo=repo,
                    endpoint="repo_info",
                    connect_timeout=args.connect_timeout,
                    read_timeout=args.read_timeout,
                    max_retries=args.retries,
                    ca_bundle=args.ca_bundle,
                    insecure=args.insecure,
                    debug_http=args.debug_http,
                    reporter=gh_reporter,
                )
                if info_result.rate_remaining is not None:
                    last_rate_remaining = info_result.rate_remaining
                if info_result.rate_reset_epoch is not None:
                    last_rate_reset_epoch = info_result.rate_reset_epoch

                gh_reporter.bump_count(classify_progress_bucket(info_result))
                for cat in summarize_result_category(info_result):
                    error_counts[cat] += 1
                    if cat != "ok":
                        add_example(error_examples, cat, repo, args.error_examples)

                info = info_result.data
                st2 = info_result.status_code
                if info and st2 < 400 and info_result.error_kind == "ok":
                    stars_total += int(info.get("stargazers_count", 0) or 0)
                    forks_total += int(info.get("forks_count", 0) or 0)
                    if info.get("archived"):
                        archived_count += 1

            gh_reporter.mark_repo_done(i)
    finally:
        gh_reporter.mark_repo_done(len(repos_list))
        gh_reporter.stop()

    print("=== GitHub stats ===")
    print(f"Repos checked: {len(repos_list)}")
    print(f"OK:           {ok}")
    print(f"404:          {not_found}")
    print(f"403:          {forbidden}")
    if other_err:
        print(f"Other:        {dict(other_err)}")
    if last_rate_remaining is not None or last_rate_reset_epoch is not None:
        print(f"RateLimit:    remaining={last_rate_remaining} reset_epoch={last_rate_reset_epoch}")
    print()

    print("GitHub request outcome summary:")
    summary_keys = [
        "ok",
        "404",
        "403",
        "rate_limited",
        "timeouts",
        "ssl_errors",
        "connect_errors",
        "5xx",
        "other_4xx",
        "json_errors",
        "forbidden_not_authorized",
        "other",
    ]
    for key in summary_keys:
        print(f"  {key:24s} {error_counts.get(key, 0):6d}")
    print()

    print(f"Error examples (up to {args.error_examples} repos/category):")
    shown_examples = 0
    for key in summary_keys:
        if key == "ok":
            continue
        examples = error_examples.get(key, [])
        if not examples:
            continue
        shown_examples += 1
        print(f"  {key:24s} {', '.join(examples)}")
    if shown_examples == 0:
        print("  (none)")
    print()

    print(f"Top-{args.top} primary languages (by repo count):")
    for lang, cnt in primary_lang_counter.most_common(args.top):
        print(f"  {lang:20s} {cnt}")
    print()

    print(f"Top-{args.top} languages by total bytes (sum over repos):")
    for lang, b in lang_bytes_total.most_common(args.top):
        print(f"  {lang:20s} {b}")
    print()

    print("Java share buckets (from GitHub /languages per repo):")
    for key in [">=75%", "50-75%", "25-50%", "10-25%", "0-10%", "0%"]:
        if key in java_share_bucket_gh:
            print(f"  {key:8s}: {java_share_bucket_gh[key]}")
    print()

    if args.repo_info:
        print("Repo metadata (from /repos endpoint):")
        print(f"  Total stars (sum): {stars_total}")
        print(f"  Total forks (sum): {forks_total}")
        print(f"  Archived repos:    {archived_count}")
        print()


if __name__ == "__main__":
    main()
