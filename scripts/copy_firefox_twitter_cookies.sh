#!/usr/bin/env bash
set -euo pipefail

# Copy Twitter/X cookies from Firefox (Snap profile) into ./cookies.json
# in a Twikit-compatible format: {"cookie_name": "cookie_value", ...}

PROFILE_ROOT="${FIREFOX_PROFILE_ROOT:-$HOME/snap/firefox/common/.mozilla/firefox}"

if [[ ! -f "$PROFILE_ROOT/profiles.ini" ]]; then
  echo "Could not find Firefox profiles.ini at: $PROFILE_ROOT/profiles.ini" >&2
  echo "Set FIREFOX_PROFILE_ROOT to your Firefox profile directory root." >&2
  exit 1
fi

PROFILE_PATH_REL="$(awk -F= '
  /^\[Profile[0-9]+\]$/ { in_profile=1; path=""; def="" }
  in_profile && /^Path=/ { path=$2 }
  in_profile && /^Default=/ { def=$2 }
  in_profile && /^\[/{ if (path != "" && def == "1") { print path; exit } }
  END { if (path != "" && def == "1") print path }
' "$PROFILE_ROOT/profiles.ini")"

if [[ -z "${PROFILE_PATH_REL:-}" ]]; then
  echo "Could not determine default Firefox profile from profiles.ini" >&2
  exit 1
fi

PROFILE_DIR="$PROFILE_ROOT/$PROFILE_PATH_REL"
COOKIES_DB="$PROFILE_DIR/cookies.sqlite"

if [[ ! -f "$COOKIES_DB" ]]; then
  echo "Could not find cookies DB at: $COOKIES_DB" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cp "$COOKIES_DB" "$TMP_DIR/cookies.sqlite"
[[ -f "$PROFILE_DIR/cookies.sqlite-wal" ]] && cp "$PROFILE_DIR/cookies.sqlite-wal" "$TMP_DIR/cookies.sqlite-wal"
[[ -f "$PROFILE_DIR/cookies.sqlite-shm" ]] && cp "$PROFILE_DIR/cookies.sqlite-shm" "$TMP_DIR/cookies.sqlite-shm"

python3 - "$TMP_DIR/cookies.sqlite" "$PWD/cookies.json" <<'PY'
import json
import sqlite3
import sys

db_path = sys.argv[1]
out_path = sys.argv[2]

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("""
SELECT name, value, host
FROM moz_cookies
WHERE host LIKE '%twitter.com' OR host LIKE '%x.com'
""")
rows = cur.fetchall()
conn.close()

cookies = {}
for name, value, host in rows:
    # Prefer x.com cookies when duplicate names exist.
    if name not in cookies or str(host).endswith("x.com"):
        cookies[name] = value

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cookies, f, ensure_ascii=True, indent=2)

print(f"Wrote {len(cookies)} cookies to {out_path}")
PY

