#!/usr/bin/env bash
# 1-Command Release Script for DAU Voice Pro
# Usage: bash scripts/release.sh 1.1.1 "Mô tả tính năng mới"
set -euo pipefail

NEW_VER="${1:-}"
NOTES="${2:-Cập nhật DAU Voice Pro v${NEW_VER}}"

if [[ -z "$NEW_VER" ]]; then
  echo "Cách dùng: bash scripts/release.sh <version> [ghi chú]"
  echo "Ví dụ   : bash scripts/release.sh 1.1.1 \"Sân khấu LED, Slide RAG & tối ưu ổn định\""
  exit 1
fi

NEW_VER="${NEW_VER#v}"
if [[ ! "$NEW_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Phiên bản không hợp lệ (cần định dạng semver, VD: 1.1.1): $NEW_VER" >&2
  exit 1
fi

TAG="v${NEW_VER}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Fail-fast guards BEFORE mutating any files
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "Lỗi: Chỉ được phát hành từ nhánh 'main' (hiện đang ở '${CURRENT_BRANCH:-detached HEAD}')." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Lỗi: Git working tree đang có thay đổi chưa commit. Hãy commit hoặc stash trước khi phát hành." >&2
  git status --short >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "Lỗi: Tag cục bộ '${TAG}' đã tồn tại. Hãy tăng số phiên bản mới." >&2
  exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "Lỗi: Tag '${TAG}' đã tồn tại trên remote 'origin'. Không ghi đè tag đã phát hành." >&2
  exit 1
fi

# 2. Bump ALL version locations atomically
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
cargo_txt, n_cargo = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{ver}"', cargo_txt, count=1)
assert n_cargo == 1, "Failed to update src-tauri/Cargo.toml version"
cargo_p.write_text(cargo_txt, encoding="utf-8")

pyproj_p = Path("engine/pyproject.toml")
pyproj_txt = pyproj_p.read_text(encoding="utf-8")
pyproj_txt, n_pyproj = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{ver}"', pyproj_txt, count=1)
assert n_pyproj == 1, "Failed to update engine/pyproject.toml version"
pyproj_p.write_text(pyproj_txt, encoding="utf-8")

server_p = Path("engine/server.py")
server_txt = server_p.read_text(encoding="utf-8")
server_txt, n_srv = re.subn(
    r'(FastAPI\([^)]*version=")[^"]+(")',
    rf'\g<1>{ver}\g<2>',
    server_txt,
    count=1,
)
assert n_srv == 1, "Failed to update engine/server.py FastAPI version"
server_p.write_text(server_txt, encoding="utf-8")

mm_p = Path("engine/model_manager.py")
mm_txt = mm_p.read_text(encoding="utf-8")
mm_txt, n_mm = re.subn(
    r'("User-Agent"\s*:\s*"PolyVoicePro/)[^"]+(")',
    rf'\g<1>{ver}\g<2>',
    mm_txt,
    count=1,
)
assert n_mm == 1, "Failed to update engine/model_manager.py User-Agent version"
mm_p.write_text(mm_txt, encoding="utf-8")

app_p = Path("src/App.tsx")
app_txt = app_p.read_text(encoding="utf-8")
app_txt, n_app = re.subn(
    r'(const\s+APP_VERSION\s*=\s*typeof\s+__APP_VERSION__\s*!==\s*"undefined"\s*\?\s*__APP_VERSION__\s*:\s*")[^"]+(")',
    rf'\g<1>{ver}\g<2>',
    app_txt,
    count=1,
)
assert n_app == 1, "Failed to update src/App.tsx APP_VERSION fallback"
app_p.write_text(app_txt, encoding="utf-8")

print(
    "Updated package.json, src-tauri/tauri.conf.json, src-tauri/Cargo.toml, "
    f"engine/pyproject.toml, engine/server.py, engine/model_manager.py, src/App.tsx -> v{ver}"
)
PY

# 3. Keep src-tauri/Cargo.lock and engine/uv.lock in sync (required for CI --frozen & parity check)
if command -v cargo >/dev/null 2>&1; then
  (cd src-tauri && cargo update --offline -p voicerecore >/dev/null 2>&1 || cargo update -p voicerecore >/dev/null 2>&1)
else
  python3 - "$NEW_VER" <<'PY'
import re, sys
from pathlib import Path

ver = sys.argv[1]
lock_p = Path("src-tauri/Cargo.lock")
txt = lock_p.read_text(encoding="utf-8")
txt, n = re.subn(
    r'(name\s*=\s*"voicerecore"\r?\nversion\s*=\s*")[^"]+(")',
    rf'\g<1>{ver}\g<2>',
    txt,
    count=1,
)
assert n == 1, "Failed to sync src-tauri/Cargo.lock"
lock_p.write_text(txt, encoding="utf-8")
PY
fi

if command -v uv >/dev/null 2>&1; then
  (cd engine && uv lock --offline >/dev/null 2>&1 || uv lock >/dev/null 2>&1)
else
  python3 - "$NEW_VER" <<'PY'
import re, sys
from pathlib import Path

ver = sys.argv[1]
lock_p = Path("engine/uv.lock")
if lock_p.exists():
    txt = lock_p.read_text(encoding="utf-8")
    txt, n = re.subn(
        r'(name\s*=\s*"engine"\r?\nversion\s*=\s*")[^"]+(")',
        rf'\g<1>{ver}\g<2>',
        txt,
        count=1,
    )
    if n == 1:
        lock_p.write_text(txt, encoding="utf-8")
PY
fi

# 4. Final parity check before committing (compatible with macOS stock Python 3.9+)
python3 - "$NEW_VER" <<'PY'
import json, re, sys
from pathlib import Path

ver = sys.argv[1]

pkg_ver = json.loads(Path("package.json").read_text(encoding="utf-8"))["version"]
assert pkg_ver == ver, f"package.json mismatch: {pkg_ver} != {ver}"

tauri_ver = json.loads(Path("src-tauri/tauri.conf.json").read_text(encoding="utf-8"))["version"]
assert tauri_ver == ver, f"tauri.conf.json mismatch: {tauri_ver} != {ver}"

cargo_m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', Path("src-tauri/Cargo.toml").read_text(encoding="utf-8"))
assert cargo_m and cargo_m.group(1) == ver, f"Cargo.toml mismatch: {cargo_m.group(1) if cargo_m else None} != {ver}"

lock_matches = re.findall(
    r'name\s*=\s*"voicerecore"\r?\nversion\s*=\s*"([^"]+)"',
    Path("src-tauri/Cargo.lock").read_text(encoding="utf-8"),
)
assert lock_matches == [ver], f"Cargo.lock mismatch: {lock_matches} != [{ver}]"

pyproj_m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', Path("engine/pyproject.toml").read_text(encoding="utf-8"))
assert pyproj_m and pyproj_m.group(1) == ver, f"engine/pyproject.toml mismatch: {pyproj_m.group(1) if pyproj_m else None} != {ver}"

uv_lock_p = Path("engine/uv.lock")
if uv_lock_p.exists():
    uv_matches = re.findall(
        r'name\s*=\s*"engine"\r?\nversion\s*=\s*"([^"]+)"',
        uv_lock_p.read_text(encoding="utf-8"),
    )
    assert uv_matches == [ver], f"engine/uv.lock mismatch: {uv_matches} != [{ver}]"

print(f"Verified version parity across all manifests & lockfiles -> v{ver}")
PY

if command -v uv >/dev/null 2>&1; then
  (cd engine && uv lock --check)
fi

git add \
  package.json \
  src-tauri/tauri.conf.json \
  src-tauri/Cargo.toml \
  src-tauri/Cargo.lock \
  engine/pyproject.toml \
  engine/server.py \
  engine/model_manager.py \
  src/App.tsx

if git diff --name-only -- engine/uv.lock | grep -q "uv.lock"; then
  git add engine/uv.lock
fi

git commit -m "release: ${TAG} — ${NOTES}"
git tag -a "${TAG}" -m "DAU Voice Pro ${TAG}: ${NOTES}"
git push origin main --follow-tags

echo ""
echo "✓ Đã đẩy ${TAG} lên GitHub! GitHub Actions đang tự build Windows .exe + macOS .dmg + latest.json."
echo "  Theo dõi tại: https://github.com/niitbeo/PolyVoice-Pro/actions"
