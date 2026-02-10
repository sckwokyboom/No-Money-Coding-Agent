#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

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


def iter_jsonl(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


PATCH_PATTERNS = {
    "java_ext": re.compile(r"\.java\b", re.IGNORECASE),
    "java_paths": re.compile(r"(?:^|\b)(src/main/java|src/test/java)\b", re.IGNORECASE),
    "maven_gradle_files": re.compile(
        r"\b(pom\.xml|build\.gradle(?:\.kts)?|settings\.gradle(?:\.kts)?|gradle\.properties)\b",
        re.IGNORECASE,
    ),
    "mvnw_gradlew": re.compile(r"\b(mvnw|gradlew)\b", re.IGNORECASE),
    "spring_word": re.compile(r"\bspring\b", re.IGNORECASE),
    "springframework": re.compile(r"\borg\.springframework\b", re.IGNORECASE),
    "boot_starter": re.compile(r"\bspring-boot-starter\b", re.IGNORECASE),
    "boot_plugin": re.compile(r"\borg\.springframework\.boot\b", re.IGNORECASE),
    "spring_annotations": re.compile(
        r"@RestController|@Controller|@Service|@Repository|@Component|@SpringBootApplication"
        r"|@RequestMapping|@GetMapping|@PostMapping|@PutMapping|@DeleteMapping",
        re.IGNORECASE,
    ),
    "spring_config": re.compile(r"\b(application\.ya?ml|application\.properties)\b", re.IGNORECASE),
}

PATCH_WEIGHTS = {
    "java_ext": 2,
    "java_paths": 3,
    "maven_gradle_files": 2,
    "mvnw_gradlew": 1,
    "spring_word": 1,
    "springframework": 2,
    "boot_starter": 4,
    "boot_plugin": 3,
    "spring_annotations": 3,
    "spring_config": 1,
}

PATCH_SCORE_CAP = 60
REQUIRED_FIELDS = ("event_id", "agent", "repo", "sha", "description", "patch")


@dataclass
class RepoSignals:
    java_share: float
    has_pom: bool
    has_gradle: bool
    has_spring_dependency: bool
    fetched_at: int


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
        "User-Agent": "nmc-agent-spring-filter",
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


def gh_repo_java_share(owner: str, repo: str, headers: Dict[str, str], *, timeout: int, max_retries: int, ca_bundle: str, insecure: bool) -> float:
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    data, status = gh_get_json(url, headers, timeout=timeout, max_retries=max_retries, ca_bundle=ca_bundle, insecure=insecure)
    if not data or status >= 400:
        return 0.0
    total = sum(v for v in data.values() if isinstance(v, (int, float)))
    if total <= 0:
        return 0.0
    java_bytes = float(data.get("Java", 0) or 0)
    return max(0.0, min(1.0, java_bytes / float(total)))


def gh_exists_path(owner: str, repo: str, path: str, headers: Dict[str, str], *, timeout: int, max_retries: int, ca_bundle: str, insecure: bool) -> Optional[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    data, status = gh_get_json(url, headers, timeout=timeout, max_retries=max_retries, ca_bundle=ca_bundle, insecure=insecure)
    if not data or status >= 400:
        return None
    return data


def decode_content_from_contents_api(obj: Dict[str, Any]) -> str:
    content_b64 = obj.get("content")
    if not content_b64 or not isinstance(content_b64, str):
        return ""
    try:
        raw = base64.b64decode(content_b64, validate=False)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def gh_repo_signals(
    owner: str,
    repo: str,
    headers: Dict[str, str],
    *,
    timeout: int,
    max_retries: int,
    ca_bundle: str,
    insecure: bool,
) -> RepoSignals:
    java_share = gh_repo_java_share(owner, repo, headers, timeout=timeout, max_retries=max_retries, ca_bundle=ca_bundle, insecure=insecure)

    pom_obj = gh_exists_path(owner, repo, "pom.xml", headers, timeout=timeout, max_retries=max_retries, ca_bundle=ca_bundle, insecure=insecure)
    gradle_obj = gh_exists_path(owner, repo, "build.gradle", headers, timeout=timeout, max_retries=max_retries, ca_bundle=ca_bundle, insecure=insecure)
    if gradle_obj is None:
        gradle_obj = gh_exists_path(owner, repo, "build.gradle.kts", headers, timeout=timeout, max_retries=max_retries, ca_bundle=ca_bundle, insecure=insecure)

    has_pom = pom_obj is not None
    has_gradle = gradle_obj is not None

    has_spring_dependency = False
    for obj in (pom_obj, gradle_obj):
        if not obj:
            continue
        text = decode_content_from_contents_api(obj)
        if re.search(r"spring-boot-starter|org\.springframework\.boot|org\.springframework", text, re.IGNORECASE):
            has_spring_dependency = True
            break

    return RepoSignals(
        java_share=java_share,
        has_pom=has_pom,
        has_gradle=has_gradle,
        has_spring_dependency=has_spring_dependency,
        fetched_at=int(time.time()),
    )


def open_cache(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_cache (
          repo TEXT PRIMARY KEY,
          java_share REAL NOT NULL,
          has_pom INTEGER NOT NULL,
          has_gradle INTEGER NOT NULL,
          has_spring_dependency INTEGER NOT NULL,
          fetched_at INTEGER NOT NULL
        )
        """
    )
    return con


def cache_get(con: sqlite3.Connection, repo_key: str, max_age_seconds: int) -> Optional[RepoSignals]:
    row = con.execute(
        "SELECT java_share, has_pom, has_gradle, has_spring_dependency, fetched_at FROM repo_cache WHERE repo = ?",
        (repo_key,),
    ).fetchone()
    if not row:
        return None
    java_share, has_pom, has_gradle, has_spring_dependency, fetched_at = row
    if int(time.time()) - int(fetched_at) > max_age_seconds:
        return None
    return RepoSignals(
        java_share=float(java_share),
        has_pom=bool(has_pom),
        has_gradle=bool(has_gradle),
        has_spring_dependency=bool(has_spring_dependency),
        fetched_at=int(fetched_at),
    )


def cache_put(con: sqlite3.Connection, repo_key: str, signals: RepoSignals) -> None:
    con.execute(
        """
        INSERT INTO repo_cache(repo, java_share, has_pom, has_gradle, has_spring_dependency, fetched_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(repo) DO UPDATE SET
          java_share=excluded.java_share,
          has_pom=excluded.has_pom,
          has_gradle=excluded.has_gradle,
          has_spring_dependency=excluded.has_spring_dependency,
          fetched_at=excluded.fetched_at
        """,
        (
            repo_key,
            signals.java_share,
            int(signals.has_pom),
            int(signals.has_gradle),
            int(signals.has_spring_dependency),
            signals.fetched_at,
        ),
    )
    con.commit()


def patch_score(patch: str) -> Tuple[int, Dict[str, int]]:
    if not patch:
        return 0, {}
    hits: Dict[str, int] = {}
    score = 0
    for key, rx in PATCH_PATTERNS.items():
        n = len(rx.findall(patch))
        if n:
            hits[key] = n
            score += PATCH_WEIGHTS.get(key, 0) * n
    return min(score, PATCH_SCORE_CAP), hits


def final_score(patch_s: int, repo_s: Optional[RepoSignals]) -> Tuple[int, Dict[str, Any]]:
    details: Dict[str, Any] = {"patch_score": patch_s}
    score = patch_s

    if repo_s is None:
        return min(score, 100), details

    details.update(
        {
            "repo_java_share": repo_s.java_share,
            "repo_has_pom": repo_s.has_pom,
            "repo_has_gradle": repo_s.has_gradle,
            "repo_has_spring_dependency": repo_s.has_spring_dependency,
            "repo_fetched_at": repo_s.fetched_at,
        }
    )

    if repo_s.java_share >= 0.50:
        score += 18
    elif repo_s.java_share >= 0.25:
        score += 10
    elif repo_s.java_share >= 0.10:
        score += 5

    if repo_s.has_pom or repo_s.has_gradle:
        score += 6

    if repo_s.has_spring_dependency:
        score += 18

    return min(score, 100), details


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_input_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")

    row = {k: raw.get(k) for k in REQUIRED_FIELDS}
    row["description"] = row.get("description") or ""
    row["patch"] = row.get("patch") or ""
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Filter claude.jsonl to Spring-related samples.")
    ap.add_argument("--config", default="", help="YAML config path")

    ap.add_argument("--input", default="", help="Path to input claude.jsonl")
    ap.add_argument("--output", default="", help="Path to output spring_only.jsonl")
    ap.add_argument("--out-all", default="", help="Optional path to save all scored rows")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means no limit")
    ap.add_argument("--include-raw", action="store_true", help="Include original JSON row in output")

    ap.add_argument("--patch-threshold", type=int, default=10)
    ap.add_argument("--final-threshold", type=int, default=40)

    ap.add_argument("--github", action="store_true", help="Enable GitHub repo signals")
    ap.add_argument("--cache-db", default="data/repo_cache.sqlite")
    ap.add_argument("--cache-max-age-hours", type=int, default=24 * 7)
    ap.add_argument("--token", default="")
    ap.add_argument("--token-file", default="")

    ap.add_argument("--ca-bundle", default="", help="Path to trusted CA bundle")
    ap.add_argument("--insecure", action="store_true", help="Disable TLS verification (not recommended)")

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
    if not args.output:
        args.output = str(deep_get(cfg, ["output", "spring_only"], "") or "")
    if not args.out_all:
        args.out_all = str(deep_get(cfg, ["output", "all_scored"], "") or "")

    args.max_rows = int(deep_get(cfg, ["max_rows"], args.max_rows))

    args.patch_threshold = int(deep_get(cfg, ["thresholds", "patch_min"], args.patch_threshold))
    args.final_threshold = int(deep_get(cfg, ["thresholds", "final_min"], args.final_threshold))

    gh_enabled = deep_get(cfg, ["github", "enabled"], None)
    if gh_enabled is not None and not args.github:
        args.github = bool(gh_enabled)

    args.cache_db = str(deep_get(cfg, ["github", "cache_db"], args.cache_db))
    args.cache_max_age_hours = int(deep_get(cfg, ["github", "cache_max_age_hours"], args.cache_max_age_hours))

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
    if not args.output:
        raise SystemExit("You must provide --output (or set output.spring_only in --config).")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {args.input}")

    ensure_parent(args.output)
    if args.out_all:
        ensure_parent(args.out_all)
    if args.github:
        ensure_parent(args.cache_db)

    total_rows = count_nonempty_lines(args.input)
    if args.max_rows and args.max_rows > 0:
        total_rows = min(total_rows, args.max_rows)

    token = load_token(args.token, args.token_file)
    headers = gh_headers(token)

    con = open_cache(args.cache_db) if args.github else None
    cache_max_age_seconds = args.cache_max_age_hours * 3600

    progress = Progress(enabled=args.progress)

    seen = 0
    valid = 0
    spring_written = 0
    invalid_json = 0
    invalid_schema = 0
    gh_ok = 0
    gh_fail = 0

    all_handle = open(args.out_all, "w", encoding="utf-8") if args.out_all else None
    spring_handle = open(args.output, "w", encoding="utf-8")

    try:
        with open(args.input, "r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue

                seen += 1
                if args.max_rows and seen > args.max_rows:
                    break

                try:
                    raw = json.loads(line)
                except Exception:
                    invalid_json += 1
                    if args.progress and (seen % args.progress_every == 0):
                        progress.update(
                            "filter",
                            seen,
                            total_rows,
                            extra=f"valid={valid} spring={spring_written} invalid={invalid_json + invalid_schema}",
                        )
                    continue

                try:
                    row = normalize_input_row(raw)
                except Exception:
                    invalid_schema += 1
                    if args.progress and (seen % args.progress_every == 0):
                        progress.update(
                            "filter",
                            seen,
                            total_rows,
                            extra=f"valid={valid} spring={spring_written} invalid={invalid_json + invalid_schema}",
                        )
                    continue

                valid += 1
                pscore, phits = patch_score(str(row["patch"]))

                repo_sig: Optional[RepoSignals] = None
                if args.github:
                    owner_repo = parse_owner_repo(str(row["repo"]))
                    if owner_repo is not None:
                        owner, repo_name = owner_repo
                        repo_key = f"{owner}/{repo_name}"
                        cached = cache_get(con, repo_key, cache_max_age_seconds) if con else None
                        if cached is not None:
                            repo_sig = cached
                        else:
                            try:
                                repo_sig = gh_repo_signals(
                                    owner,
                                    repo_name,
                                    headers,
                                    timeout=args.timeout,
                                    max_retries=args.retries,
                                    ca_bundle=args.ca_bundle,
                                    insecure=args.insecure,
                                )
                                gh_ok += 1
                                if con:
                                    cache_put(con, repo_key, repo_sig)
                            except Exception:
                                gh_fail += 1

                fscore, fdetails = final_score(pscore, repo_sig)

                out = {
                    "event_id": row["event_id"],
                    "agent": row["agent"],
                    "repo": row["repo"],
                    "sha": row["sha"],
                    "description": row["description"],
                    "patch": row["patch"],
                    "patch_score": pscore,
                    "patch_hits": phits,
                    "final_score": fscore,
                    "signals": fdetails,
                }
                if args.include_raw:
                    out["raw"] = raw

                if all_handle is not None:
                    all_handle.write(json.dumps(out, ensure_ascii=False) + "\n")

                if pscore >= args.patch_threshold and fscore >= args.final_threshold:
                    spring_handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                    spring_written += 1

                if args.progress and (seen % args.progress_every == 0 or seen == total_rows):
                    progress.update(
                        "filter",
                        seen,
                        total_rows,
                        extra=(
                            f"valid={valid} spring={spring_written} invalid={invalid_json + invalid_schema} "
                            f"gh_ok={gh_ok} gh_fail={gh_fail}"
                        ),
                    )

        if args.progress:
            progress.update(
                "filter",
                min(seen, total_rows),
                total_rows,
                extra=(
                    f"valid={valid} spring={spring_written} invalid={invalid_json + invalid_schema} "
                    f"gh_ok={gh_ok} gh_fail={gh_fail}"
                ),
                force=True,
            )
            progress.done()

    finally:
        if con:
            con.close()
        spring_handle.close()
        if all_handle is not None:
            all_handle.close()

    print(f"input: {args.input}")
    print(f"total_seen: {seen}")
    print(f"valid_rows: {valid}")
    print(f"invalid_json: {invalid_json}")
    print(f"invalid_schema: {invalid_schema}")
    print(f"spring_only: {spring_written}")
    print(f"output: {args.output}")
    if args.out_all:
        print(f"out_all: {args.out_all}")
    if args.github:
        print(f"cache_db: {args.cache_db}")


if __name__ == "__main__":
    main()
