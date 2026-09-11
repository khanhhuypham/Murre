"""dataset/vitext2sql.py — Đọc ViText2SQL THÔ theo cách MURRE hiểu được.

ViText2SQL giữ nguyên định dạng thô của Spider, còn MURRE cần bản đã tiền xử lý.
Chỗ lệch nằm ở ba khoá:

    tables.json   THIẾU khoá `schema` — danh sách chuỗi "db.bảng(cột, ...)" mà
                  retrieval encode. Dựng từ table_names + column_names.

    dev.json      thô có: db_id, query, query_toks, question, question_toks, sql
                  MURRE cần: id, db_id, utterance, query, rel_schema
                  `rel_schema` (bảng gold) không có sẵn, phải suy từ câu truy vấn.

VÌ SAO THÍCH NGHI TRONG CODE, KHÔNG GHI ĐÈ FILE
-----------------------------------------------
File dưới dataset/vitext2sql/data/ là bản sao NGUYÊN XI của upstream, so được
từng byte với bản gốc của VinAI. Ghi đè chúng bằng bản đã đổi khoá là mất khả năng
đó: không còn biết dữ liệu đang chạy có đúng dữ liệu gốc hay không, và tải lại
(bằng tay) là mất hết phần đã chuyển đổi.

Chi phí: mỗi lần đọc phải dựng lại `schema` và `rel_schema`. Đo trên split dev
(954 câu, 166 database) là dưới một giây — không đáng kể so với việc mã hoá
corpus, mà corpus thì đã có cache riêng.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# =============================================================================
# tables.json → thêm khoá `schema`
# =============================================================================
def schema_strings(db: Dict[str, Any]) -> List[str]:
    """Một chuỗi "db_id.bảng(cột, cột, ...)" cho mỗi bảng — đầu vào của retrieval.

    Đúng quy ước của dataset/spider/tables.json: dùng tên đã chuẩn hoá
    (table_names / column_names, không phải *_original), bỏ cột ảo "*".
    """
    db_id: str = db["db_id"]
    columns: List[List[str]] = [[] for _ in db["table_names"]]
    for table_idx, column_name in db["column_names"]:
        if table_idx >= 0:  # -1 là cột ảo "*", không thuộc bảng nào
            columns[table_idx].append(column_name)

    return [
        f"{db_id}.{table_name}({', '.join(columns[i])})"
        for i, table_name in enumerate(db["table_names"])
    ]


def adapt_tables(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """tables.json thô → bản có thêm khoá `schema`. KHÔNG sửa dict gốc."""
    return [{**db, "schema": schema_strings(db=db)} for db in raw]


# =============================================================================
# Suy ra bảng gold của một câu hỏi
# =============================================================================
def tables_from_tokens(
    query_toks: Sequence[str], table_names: Sequence[str],
) -> List[str]:
    """Bảng đứng ngay sau FROM / JOIN trong câu truy vấn, theo thứ tự xuất hiện.

    Đọc `query_toks` chứ không regex trên chuỗi: tên bảng tiếng Việt có dấu cách
    ("kiến trúc sư") nhưng ViText2SQL vẫn giữ nó thành MỘT token, nên so khớp
    token là chính xác, không phải đoán biên từ.

    FROM mở subquery thì token kế là "(" — không khớp tên bảng nào, và bảng bên
    trong subquery do tables_from_ast() nhặt.
    """
    known: Dict[str, str] = {name.lower(): name for name in table_names}
    toks: List[str] = [t.lower() for t in query_toks]

    out: List[str] = []
    for i, tok in enumerate(toks[:-1]):
        if tok in ("from", "join"):
            name: Optional[str] = known.get(toks[i + 1])
            if name is not None and name not in out:
                out.append(name)
    return out


def tables_from_ast(node: Any, found: Optional[List[int]] = None) -> List[int]:
    """Mọi ["table_unit", <chỉ số>] trong cây `sql`, kể cả trong subquery.

    Duyệt đệ quy vì bảng gold không chỉ nằm ở FROM ngoài cùng mà còn ở subquery
    trong WHERE/HAVING và ở các nhánh INTERSECT/UNION/EXCEPT.
    """
    if found is None:
        found = []

    if isinstance(node, dict):
        for value in node.values():
            tables_from_ast(node=value, found=found)
    elif isinstance(node, list):
        if len(node) == 2 and node[0] == "table_unit" and isinstance(node[1], int):
            if node[1] not in found:
                found.append(node[1])
            return found
        for item in node:
            tables_from_ast(node=item, found=found)

    return found


def gold_tables(row: Dict[str, Any], db: Dict[str, Any]) -> List[str]:
    """Tên các bảng gold của một câu hỏi = HỢP của hai cách suy ra.

    Quy tắc "mọi bảng trong FROM/JOIN, kể cả subquery" đã đối chiếu với
    dataset/spider/dev.json: khớp tập bảng 658/658 câu.

    Phải dùng CẢ HAI nguồn vì mỗi nguồn thiếu một kiểu:
      - Cây `sql` của ViText2SQL BỎ SÓT bảng thứ ba ở câu JOIN 3 bảng (44/954 câu
        của split dev) — lấy mình nó thì gold thiếu bảng và recall đo ra sai.
      - Token chỉ thấy bảng viết thẳng sau FROM/JOIN, không thấy bảng nằm trong
        subquery ở mệnh đề FROM.
    """
    names: List[str] = list(db["table_names"])
    out: List[str] = tables_from_tokens(query_toks=row["query_toks"], table_names=names)

    for idx in tables_from_ast(node=row["sql"]):
        if idx < len(names) and names[idx] not in out:
            out.append(names[idx])
    return out


# =============================================================================
# split thô (dev/test/train) → record MURRE đọc được
# =============================================================================
def adapt_split(
    raw: List[Dict[str, Any]], tables: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """dev.json thô → record có `utterance` và `rel_schema`.

    `tables` phải là bản ĐÃ qua adapt_tables() (cần khoá `schema`).
    """
    dbs: Dict[str, Dict[str, Any]] = {db["db_id"]: db for db in tables}

    missing: List[str] = sorted({r["db_id"] for r in raw} - set(dbs))
    if missing:
        raise ValueError(
            f"{len(missing)} db_id có trong split nhưng không có trong tables.json: "
            f"{missing[:5]}. Hai file phải cùng một mức tách từ "
            f"(syllable-level hay word-level) — xem datasets.vitext2sql trong config.yaml."
        )

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(raw):
        db: Dict[str, Any] = dbs[row["db_id"]]
        # schema[j] ứng với table_names[j] — schema_strings() dựng theo đúng thứ tự đó.
        by_name: Dict[str, str] = dict(zip(db["table_names"], db["schema"]))

        out.append({
            "id": i,
            "db_id": row["db_id"],
            "utterance": row["question"],
            "query": row["query"],
            "rel_schema": [by_name[n] for n in gold_tables(row=row, db=db)],
        })
    return out
