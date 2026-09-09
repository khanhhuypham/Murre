"""scripts/download_vitext2sql.py — Tải ViText2SQL về, GIỮ NGUYÊN XI dữ liệu gốc.

    python scripts/download_vitext2sql.py                  # tải cả 2 mức, mọi split
    python scripts/download_vitext2sql.py --level syllable  # chỉ 1 mức
    python scripts/download_vitext2sql.py --verify          # chỉ kiểm tra, không tải
    python scripts/download_vitext2sql.py --force           # tải lại cả file đã có

Nguồn: https://github.com/VinAIResearch/ViText2SQL (bản dịch Spider sang tiếng Việt).

KHÔNG ĐỘNG VÀO DỮ LIỆU
----------------------
File tải về được ghi Y NGUYÊN, không parse lại, không format lại, không thêm bớt
khoá nào. Cây thư mục cũng sao chép đúng upstream:

    dataset/vitext2sql/
    ├── MANIFEST.json      ← script ghi ra, NẰM NGOÀI cây dữ liệu
    └── data/              ← bản sao nguyên vẹn của ViText2SQL/data/
        ├── syllable-level/{dev,test,train,tables}.json, test_gold.sql
        └── word-level/    (như trên)

MURRE cần định dạng khác (tables.json phải có khoá `schema`, dev.json phải có
`utterance` + `rel_schema`) — phần thích nghi đó nằm trong CODE, ở
dataset/vitext2sql.py, chạy lúc đọc file. Nhờ vậy dữ liệu trên đĩa luôn đối chiếu
được với upstream.

KIỂM TRA TOÀN VẸN
-----------------
Mỗi file được đối chiếu với GitHub API bằng hai thứ: kích thước, và git blob SHA-1
(= sha1("blob <độ dài>\\0" + nội dung) — đúng cách git tự băm). Khớp cả hai nghĩa
là file trên đĩa giống upstream từng byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO: str = "VinAIResearch/ViText2SQL"
API_URL: str = f"https://api.github.com/repos/{REPO}/contents/data/{{level}}-level"
RAW_URL: str = f"https://raw.githubusercontent.com/{REPO}/master/data/{{level}}-level/{{name}}"

LEVELS: Tuple[str, ...] = ("syllable", "word")
DEFAULT_OUT: str = "dataset/vitext2sql"

# Tải theo khối để không nuốt trọn 27MB vào RAM một lúc.
_CHUNK: int = 1 << 20


def git_blob_sha(data: bytes) -> str:
    """Băm y hệt cách git băm một file — đối chiếu được thẳng với `sha` của API."""
    header: bytes = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def list_remote(level: str) -> List[Dict[str, Any]]:
    """Danh sách file của một mức trên GitHub, kèm size và sha để đối chiếu."""
    url: str = API_URL.format(level=level)
    with urllib.request.urlopen(url, timeout=60) as resp:
        entries: List[Dict[str, Any]] = json.loads(resp.read().decode("utf-8"))
    return [e for e in entries if e.get("type") == "file"]


def download(level: str, name: str) -> bytes:
    """Tải một file về dưới dạng BYTES THÔ — không decode, không parse."""
    url: str = RAW_URL.format(level=level, name=name)
    chunks: List[bytes] = []
    with urllib.request.urlopen(url, timeout=600) as resp:
        while True:
            chunk: bytes = resp.read(_CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def read_local(path: str) -> Optional[bytes]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def check(data: bytes, entry: Dict[str, Any]) -> Optional[str]:
    """None = khớp upstream; ngược lại trả về câu mô tả chỗ lệch."""
    if len(data) != entry["size"]:
        return f"kích thước {len(data):,} ≠ upstream {entry['size']:,}"
    actual: str = git_blob_sha(data)
    if actual != entry["sha"]:
        return f"sha {actual[:12]} ≠ upstream {entry['sha'][:12]}"
    return None


def _human(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def sync_level(
    level: str, out_dir: str, verify_only: bool, force: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Đồng bộ một mức. Trả về (mục cho manifest, danh sách lỗi)."""
    entries: List[Dict[str, Any]] = list_remote(level=level)
    target: str = os.path.join(out_dir, "data", f"{level}-level")
    os.makedirs(target, exist_ok=True)

    records: List[Dict[str, Any]] = []
    problems: List[str] = []

    for entry in entries:
        name: str = entry["name"]
        path: str = os.path.join(target, name)
        local: Optional[bytes] = read_local(path=path)

        # File đã có và còn khớp upstream thì thôi, khỏi tải lại 27MB.
        if local is not None and not force and check(data=local, entry=entry) is None:
            print(f"  = {level:8s} {name:16s} {_human(entry['size']):>8s}  đã khớp")
            records.append({"path": f"data/{level}-level/{name}",
                            "size": entry["size"], "sha": entry["sha"]})
            continue

        if verify_only:
            reason: str = "chưa có" if local is None else check(data=local, entry=entry)
            problems.append(f"{level}-level/{name}: {reason}")
            print(f"  ! {level:8s} {name:16s} {_human(entry['size']):>8s}  {reason}")
            continue

        print(f"  ↓ {level:8s} {name:16s} {_human(entry['size']):>8s}  đang tải ...", flush=True)
        data: bytes = download(level=level, name=name)

        mismatch: Optional[str] = check(data=data, entry=entry)
        if mismatch is not None:
            # KHÔNG ghi file hỏng ra đĩa: thà thiếu file còn hơn có file sai mà
            # tưởng là đúng.
            problems.append(f"{level}-level/{name}: tải về bị lệch — {mismatch}")
            print(f"    HỎNG: {mismatch}")
            continue

        # Ghi nhị phân, nguyên xi. Không json.load rồi json.dump — làm vậy là đổi
        # thụt lề, thứ tự khoá, cách escape unicode; file sẽ không còn khớp sha.
        with open(path, "wb") as f:
            f.write(data)

        records.append({"path": f"data/{level}-level/{name}",
                        "size": entry["size"], "sha": entry["sha"]})

    return records, problems


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--level", choices=(*LEVELS, "all"), default="all",
        help="mức tách từ cần tải (mặc định: cả hai)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="thư mục đích")
    parser.add_argument(
        "--verify", action="store_true",
        help="chỉ đối chiếu file đang có với upstream, không tải gì",
    )
    parser.add_argument(
        "--force", action="store_true", help="tải lại cả file đã khớp",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    levels: List[str] = list(LEVELS) if args.level == "all" else [args.level]
    print(f"ViText2SQL — {'kiểm tra' if args.verify else 'tải'} {', '.join(levels)}-level")
    print(f"  nguồn: https://github.com/{REPO}/tree/master/data")

    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    for level in levels:
        got, bad = sync_level(
            level=level, out_dir=args.out, verify_only=args.verify, force=args.force,
        )
        records.extend(got)
        problems.extend(bad)

    if problems:
        print("\nCÓ VẤN ĐỀ:")
        for p in problems:
            print(f"  - {p}")
        if args.verify:
            print("\n  Chạy lại không có --verify để tải những file còn thiếu/lệch.")
        return 1

    if args.verify:
        print(f"\nMọi file ({len(records)}) khớp upstream từng byte.")
        return 0

    # MANIFEST nằm NGOÀI data/ để cây dữ liệu vẫn là bản sao thuần của upstream.
    manifest: str = os.path.join(args.out, "MANIFEST.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "source": f"https://github.com/{REPO}/tree/master/data",
            "note": "Dữ liệu trong data/ giữ nguyên xi upstream. "
                    "Phần thích nghi sang định dạng MURRE nằm ở dataset/vitext2sql.py.",
            "verify": "python scripts/download_vitext2sql.py --verify",
            "files": sorted(records, key=lambda r: r["path"]),
        }, f, ensure_ascii=False, indent=2)

    total: int = sum(r["size"] for r in records)
    print(f"\nXong {len(records)} file ({_human(total)}) → {os.path.join(args.out, 'data')}")
    print(f"  → {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
