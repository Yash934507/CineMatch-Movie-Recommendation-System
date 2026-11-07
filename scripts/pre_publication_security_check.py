#!/usr/bin/env python3
"""Scan the project for obvious secret, local data, or publication-risk files.

The tool reports file paths and categories only. It does not print any secret values.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    '.git',
    '.venv',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    'node_modules',
    '.next',
    '.nuxt',
    'dist',
    'build',
    'work',
    '.chart-data-D9ewcW',
    '.chart-data-j40yDQ',
    '.chart-data-QAepVt',
}

PATTERNS = {
    'api_key': [
        re.compile(r'(?i)(api[_-]?key|apikey)[\s:=\"\']+[A-Za-z0-9_\-]{8,}'),
        re.compile(r'(?i)(secret|token|password)[\s:=\"\']+[A-Za-z0-9_\-!@#$%^&*()=+./:,;~]{8,}'),
    ],
    'environment_file': [re.compile(r'(?i)^\s*(?:export\s+)?[A-Z0-9_]+\s*=\s*.*$')],
    'database_file': [re.compile(r'(?i).*\.(db|sqlite|sqlite3)$')],
    'private_url': [re.compile(r'(?i)(https?://|postgresql://|mysql://|mongodb://)[A-Za-z0-9._%+-]+@')],
    'personal_contact': [re.compile(r'(?i)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')],
    'local_path': [re.compile(r'(?i)(?:[A-Za-z]:\\|/Users/|/home/|/var/|/tmp/|C:\\Users\\)')],
}


def is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDED_DIRS) or any(part.startswith('.') and part not in {'.env.example', '.github'} for part in path.parts)


def normalize(rel: Path) -> str:
    return rel.as_posix()


def scan_file(path: Path, findings: dict[str, set[str]]) -> None:
    if path.is_dir():
        return
    if path.name == '.env.example':
        return
    if path.name in {'.gitignore', '.dockerignore'}:
        # These are allowed project files. We still inspect content for obvious secrets.
        pass

    # Skip binary and generated packaged files that are not source text.
    suffix = path.suffix.lower()
    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.pptx', '.ico', '.svg', '.db', '.sqlite3', '.pkl'}:
        if path.name.endswith(('.db', '.sqlite3')):
            findings['database_file'].add(normalize(path.relative_to(ROOT)))
        return

    relative = path.relative_to(ROOT)
    if relative.parts[:2] == ('data', 'raw') or relative.parts[:2] == ('data', 'processed'):
        if suffix in {'.csv', '.txt'}:
            return

    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return

    if path.name == '.env' or path.suffix == '.env':
        findings['environment_file'].add(normalize(relative))
    if path.name.endswith(('.db', '.sqlite3')):
        findings['database_file'].add(normalize(relative))

    for category, regexes in PATTERNS.items():
        for pattern in regexes:
            if pattern.search(text):
                findings[category].add(normalize(path.relative_to(ROOT)))
                break


def main() -> int:
    findings: dict[str, set[str]] = {name: set() for name in PATTERNS}
    findings['database_file'] = set()
    findings['environment_file'] = set()

    for path in sorted(ROOT.rglob('*')):
        if not path.exists() or path.is_symlink():
            continue
        if is_excluded(path):
            continue
        scan_file(path, findings)

    hit_any = False
    for category in ('api_key', 'environment_file', 'database_file', 'private_url', 'personal_contact', 'local_path'):
        items = sorted(findings.get(category, set()))
        if not items:
            continue
        hit_any = True
        print(f"[{category}]")
        for item in items:
            print(f"- {item}")

    if not hit_any:
        print('No obvious secret or local publication-risk patterns were detected in the repository source tree.')
    else:
        print('\nReview these items before any public release and keep only the approved, sanitized files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
