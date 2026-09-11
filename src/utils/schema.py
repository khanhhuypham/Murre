# =============================================================================
# utils/schema.py — Xử lý schema bảng SQL
#
#   build_schema_corpus()       → danh sách schema phẳng cho retrieval
#   build_db_index()            → {db_id: db_dict}
#   filter_ret_tables_from_db() → lọc DB dict chỉ giữ các bảng đã retrieve
#   pack_table()                → render DB dict thành CREATE TABLE SQL
# =============================================================================


from copy import deepcopy
from typing import Any, Dict, List


# =============================================================================
# NHÓM 1: Xây dựng corpus cho retrieval
# =============================================================================

def build_schema_corpus(tables: List[Dict[str, Any]]) -> List[str]:
    """Danh sách phẳng schema từ tables.json.

    Mỗi chuỗi dạng "db_id.table_name(col1, col2, ...)".
    """
    corpus: List[str] = []
    for db in tables:
        for schema_str in db.get("schema", []):
            corpus.append(schema_str)
    return corpus


def build_db_index(tables: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{db_id: db_dict} để tra cứu nhanh theo tên DB."""
    return {d["db_id"]: d for d in tables}


# =============================================================================
# NHÓM 2: Filter DB dict (dùng khi sinh SQL)
# =============================================================================

def _re_index(index_map: List[int], ind: int) -> int:
    """Tính lại chỉ số sau khi lọc; -1 (cột ảo "*") giữ nguyên."""
    if ind == -1:
        return -1
    return index_map.index(ind)


def filter_ret_tables_from_db(
    db_dict:        Dict[str, Any],
    db_id:          str,
    ret_tables_list: List[str],
) -> Dict[str, Any]:
    """Lọc DB dict chỉ giữ bảng trong ret_tables_list.

    Cập nhật lại chỉ số cột, khóa chính, khóa ngoại theo vị trí mới.
    """
    if db_dict is None:
        raise ValueError(f"db_dict là None với db_id={db_id}")

    ndb: Dict[str, Any] = deepcopy(x=db_dict)

    # Chỉ số các bảng cần giữ (trong table_names gốc)
    table_ids: List[int]  = [
        i for i, name in enumerate(ndb["table_names"])
        if name in ret_tables_list
    ]

    # Chỉ số các cột thuộc bảng giữ lại (hoặc cột ảo "*" có idx=-1)
    col_ids: List[int] = [
        i for i, (t_idx, _) in enumerate(ndb["column_names"])
        if t_idx in table_ids or t_idx == -1
    ]

    # Giữ cột thuộc bảng đã lọc, tính lại table_idx theo vị trí mới
    ndb["column_names"] = [
        [_re_index(index_map=table_ids, ind=t_idx), col_name]
        for t_idx, col_name in db_dict["column_names"]
        if t_idx in table_ids or t_idx == -1
    ]

    ndb["column_types"] = [
        ct for i, ct in enumerate(db_dict["column_types"])
        if i in col_ids
    ]

    ndb["column_names_original"] = [
        [_re_index(index_map=table_ids, ind=t_idx), col_name]
        for t_idx, col_name in db_dict["column_names_original"]
        if t_idx in table_ids or t_idx == -1
    ]

    # Khóa chính của bảng đã lọc, tính lại chỉ số theo col_ids mới
    ndb["primary_keys"] = [
        _re_index(index_map=col_ids, ind=pk)
        for pk in db_dict["primary_keys"]
        if pk in col_ids
    ]

    # Chỉ giữ khóa ngoại trong nội bộ bảng đã lọc
    ndb["foreign_keys"] = [
        [_re_index(index_map=col_ids, ind=from_idx), _re_index(index_map=col_ids, ind=to_idx)]
        for from_idx, to_idx in db_dict["foreign_keys"]
        if from_idx in col_ids and to_idx in col_ids
    ]

    ndb["table_names_original"] = [
        name for i, name in enumerate(db_dict["table_names_original"])
        if i in table_ids
    ]
    ndb["table_names"] = [
        name for i, name in enumerate(db_dict["table_names"])
        if i in table_ids
    ]

    return ndb


# =============================================================================
# NHÓM 3: Render CREATE TABLE SQL (dùng trong bước sinh SQL)
# =============================================================================

def pack_table(db: Dict[str, Any], use_original: bool = True) -> str:
    """
    Render DB dict thành chuỗi CREATE TABLE SQL.

    Tham số:
        db           : DB dict (đã được lọc bởi filter_ret_tables_from_db)
        use_original : True → dùng tên gốc (column_names_original)
                       False → dùng tên lowercase (column_names)

    Trả về:
        Chuỗi SQL, các bảng cách nhau bởi 2 dòng trống
    """
    col_names   = db["column_names_original"] if use_original else db["column_names"]
    col_types   = db["column_types"]
    primary_keys = db["primary_keys"]
    foreign_keys = db["foreign_keys"]
    table_names = db["table_names_original"] if use_original else db["table_names"]

    def get_columns(tidx: int) -> List[tuple]:
        """Lấy danh sách (tên cột, kiểu dữ liệu) của bảng tidx."""
        return [
            (name, col_types[i])
            for i, (t_idx, name) in enumerate(col_names)
            if t_idx == tidx
        ]

    def get_primary_keys(tidx: int) -> List[str]:
        """Lấy danh sách tên cột là khóa chính của bảng tidx."""
        return [
            name
            for i, (t_idx, name) in enumerate(col_names)
            if t_idx == tidx and i in primary_keys
        ]

    def get_foreign_keys(tidx: int) -> List[tuple]:
        """Lấy danh sách (from_col, to_table, to_col) là khóa ngoại của bảng tidx."""
        fks = []
        col_idx_in_table = [
            i for i, (t_idx, _) in enumerate(col_names) if t_idx == tidx
        ]
        for from_idx, to_idx in foreign_keys:
            if from_idx in col_idx_in_table:
                from_col   = col_names[from_idx][1]
                to_table   = table_names[col_names[to_idx][0]]
                to_col     = col_names[to_idx][1]
                fks.append((from_col, to_table, to_col))
        return fks

    # Tạo câu lệnh CREATE TABLE cho từng bảng
    statements = []
    for i, tname in enumerate(table_names):
        # Các cột với kiểu dữ liệu
        col_parts: List[str] = []
        for col_name, col_type in get_columns(i):
            sql_type = "text" if col_type == "text" else "int" if col_type == "number" else col_type
            col_parts.append(f"{col_name} {sql_type}")

        # Khóa chính
        pks: List[str] = get_primary_keys(i)
        pk_part: List[str] = [f"PRIMARY KEY ({', '.join(pks)})"] if pks else []

        # Khóa ngoại
        fk_parts: List[str] = [
            f"FOREIGN KEY ({fc}) REFERENCES {tt}({tc})"
            for fc, tt, tc in get_foreign_keys(i)
        ]

        # Ghép lại thành câu CREATE TABLE
        body:str = " ,\n".join(col_parts + pk_part + fk_parts)
        statements.append(f"CREATE TABLE {tname} (\n{body}\n);")

    # Các bảng cách nhau bởi 2 dòng trống, lowercase theo tác giả
    return "\n\n".join(statements).lower()
