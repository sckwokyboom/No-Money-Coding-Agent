#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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


def gh_get_json(
    url: str,
    headers: Dict[str, str],
    *,
    timeout: int,
    max_retries: int,
    ca_bundle: str,
    insecure: bool,
) -> Tuple[Optional[Dict[str, Any]], int]:
    verify: Any = True
    if insecure:
        verify = False
    elif ca_bundle:
        verify = ca_bundle

    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, timeout=timeout, verify=verify)

        if r.status_code == 404:
            return None, 404

        if r.status_code == 403:
            text = (r.text or "").lower()
            if "rate limit" in text or "abuse" in text:
                time.sleep(min(60, 2 ** attempt))
                continue
            return None, 403

        if r.status_code >= 500:
            time.sleep(min(30, 2 ** attempt))
            continue

        if r.status_code >= 400:
            return None, r.status_code

        try:
            return r.json(), r.status_code
        except Exception:
            return None, r.status_code

    return None, 0


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

    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=4)

    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-every", type=int, default=250)

    return ap


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
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

    args.timeout = int(deep_get(cfg, ["http", "timeout"], args.timeout))
    args.retries = int(deep_get(cfg, ["http", "retries"], args.retries))

    progress_enabled = deep_get(cfg, ["progress", "enabled"], None)
    if progress_enabled is not None and not args.progress:
        args.progress = bool(progress_enabled)
    args.progress_every = int(deep_get(cfg, ["progress", "every"], args.progress_every))

    return args


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    args = apply_config(args)

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

    gh_progress = Progress(enabled=args.progress)

    for i, repo in enumerate(repos_list, 1):
        pr = parse_owner_repo(repo)
        if pr is None:
            continue
        owner, name = pr

        languages_url = f"https://api.github.com/repos/{owner}/{name}/languages"
        langs, status = gh_get_json(
            languages_url,
            headers,
            timeout=args.timeout,
            max_retries=args.retries,
            ca_bundle=args.ca_bundle,
            insecure=args.insecure,
        )

        if status == 404:
            not_found += 1
        elif status == 403:
            forbidden += 1
        elif not langs or status >= 400:
            other_err[status] += 1
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
            info, st2 = gh_get_json(
                info_url,
                headers,
                timeout=args.timeout,
                max_retries=args.retries,
                ca_bundle=args.ca_bundle,
                insecure=args.insecure,
            )
            if info and st2 < 400:
                stars_total += int(info.get("stargazers_count", 0) or 0)
                forks_total += int(info.get("forks_count", 0) or 0)
                if info.get("archived"):
                    archived_count += 1

        if args.progress and (i % args.progress_every == 0 or i == len(repos_list)):
            gh_progress.update(
                "stats-github",
                i,
                len(repos_list),
                extra=f"ok={ok} 404={not_found} 403={forbidden} other={sum(other_err.values())}",
            )

    if args.progress:
        gh_progress.update(
            "stats-github",
            len(repos_list),
            len(repos_list),
            extra=f"ok={ok} 404={not_found} 403={forbidden} other={sum(other_err.values())}",
            force=True,
        )
        gh_progress.done()

    print("=== GitHub stats ===")
    print(f"Repos checked: {len(repos_list)}")
    print(f"OK:           {ok}")
    print(f"404:          {not_found}")
    print(f"403:          {forbidden}")
    if other_err:
        print(f"Other:        {dict(other_err)}")
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
