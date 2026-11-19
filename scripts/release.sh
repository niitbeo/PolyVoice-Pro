#!/usr/bin/env bash
# 1-Command Release Script for PolyVoice Pro
# Usage: bash scripts/release.sh 0.8.1 "Mô tả tính năng mới"
set -euo pipefail

NEW_VER="${1:-}"
NOTES="${2:-Cập nhật PolyVoice Pro v${NEW_VER}}"

if [[ -z "$NEW_VER" ]]; then
  echo "Cách dùng: bash scripts/release.sh <version> [ghi chú]"
  echo "Ví dụ   : bash scripts/release.sh 0.8.1 \"Cải thiện tốc độ dịch tiếng Thái\""
  exit 1
fi

NEW_VER="${NEW_VER#v}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - "$NEW_VER" <<'PY'
import json, re, sys
from pathlib import Path

ver = sys.argv[1]

pkg_p = Path("package.json")
pkg = json.loads(pkg_p.read_text(encoding="utf-8"))
pkg["version"] = ver
pkg_p.write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

conf_p = Path("src-tauri/tauri.conf.json")
conf = json.loads(conf_p.read_text(encoding="utf-8"))
conf["version"] = ver
conf_p.write_text(json.dumps(conf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

cargo_p = Path("src-tauri/Cargo.toml")
cargo_txt = cargo_p.read_text(encoding="utf-8")
cargo_txt = re.sub(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{ver}"', cargo_txt, count=1)
cargo_p.write_text(cargo_txt, encoding="utf-8")
print(f"Updated package.json, tauri.conf.json, Cargo.toml -> v{ver}")
PY

git add package.json src-tauri/tauri.conf.json src-tauri/Cargo.toml
git commit -m "release: v${NEW_VER} — ${NOTES}"
git tag -a "v${NEW_VER}" -m "PolyVoice Pro v${NEW_VER}: ${NOTES}"
git push origin main --follow-tags

echo ""
echo "✓ Đã đẩy v${NEW_VER} lên GitHub! GitHub Actions đang tự build Windows .exe + macOS .dmg + latest.json."
echo "  Theo dõi tại: https://github.com/niitbeo/PolyVoice-Pro/actions"
