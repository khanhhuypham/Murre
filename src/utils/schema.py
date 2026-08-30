# =============================================================================
# utils/schema.py — Các hàm xử lý schema bảng SQL
#
# Gồm 3 nhóm, đúng thứ tự xuất hiện bên dưới:
#   1. build_schema_corpus()       → danh sách schema phẳng cho retrieval
#      build_db_index()            → {db_id: db_dict} để tra nhanh khi sinh SQL
#   2. filter_ret_tables_from_db() → lọc DB dict chỉ giữ các bảng đã retrieve
#   3. pack_table()                → render DB dict thành CREATE TABLE SQL
#
# Nhóm 2 và 3 chỉ dùng ở steps/infer.py (bước sinh SQL); nhóm 1 dùng ở cả
# core/corpus.py và api/dependencies.py.
#
# Trung thành với implement của tác giả trong retrieve/utils.py
# =============================================================================


from copy import deepcopy
from typing import Any, Dict, List


# =============================================================================
# NHÓM 1: Xây dựng corpus cho retrieval
# =============================================================================

def build_schema_corpus(tables: List[Dict[str, Any]]) -> List[str]:
    """
    Trích xuất danh sách phẳng các chuỗi schema từ tables.json.

    Mỗi chuỗi có dạng: "db_id.table_name(col1, col2, ...)"
    Ví dụ: "concert_singer.singer(singer_id, name, country, age)"

    Tham số:
        tables : danh sách DB dict đọc từ tables.json

    Trả về:
        Danh sách phẳng tất cả schema strings của mọi bảng trong mọi DB
    """
    corpus: List[str] = []
    for db in tables:
        # Mỗi DB dict có trường "schema" chứa danh sách chuỗi schema đã build sẵn
        for schema_str in db.get("schema", []):
            corpus.append(schema_str)
    return corpus


def build_db_index(tables: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Tạo dictionary {db_id: db_dict} để tra cứu nhanh theo tên DB.

    Dùng trong bước sinh SQL để pack CREATE TABLE từ bảng đã retrieve.
    """
    return {d["db_id"]: d for d in tables}


# =============================================================================
# NHÓM 2: Filter DB dict (dùng khi sinh SQL)
# =============================================================================

def _re_index(index_map: List[int], ind: int) -> int:
    """
    Tính lại chỉ số sau khi lọc.
    ind = -1 (cột ảo "*") → giữ nguyên -1
    Ngược lại → tìm vị trí mới trong index_map
    """
    if ind == -1:
        return -1
    return index_map.index(ind)


def filter_ret_tables_from_db(
    db_dict:        Dict[str, Any],
    db_id:          str,
    ret_tables_list: List[str],
) -> Dict[str, Any]:
    """
    Lọc DB dict để chỉ giữ lại các bảng trong ret_tables_list.
    Đồng thời cập nhật lại chỉ số cột, khóa chính, khóa ngoại.

    Trung thành với filter_ret_tables_from_db() của tác giả trong retrieve/utils.py.

    Tham số:
        db_dict         : DB dict gốc từ tables.json
        db_id           : tên database (để log lỗi)
        ret_tables_list : danh sách tên bảng cần giữ lại

    Trả về:
        DB dict mới chỉ chứa các bảng trong ret_tables_list
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

    # Cập nhật column_names: chỉ giữ cột thuộc bảng đã lọc
    # Tính lại table_idx theo vị trí mới trong table_ids
    ndb["column_names"] = [
        [_re_index(index_map=table_ids, ind=t_idx), col_name]
        for t_idx, col_name in db_dict["column_names"]
        if t_idx in table_ids or t_idx == -1
    ]

    # Cập nhật column_types tương ứng
    ndb["column_types"] = [
        ct for i, ct in enumerate(db_dict["column_types"])
        if i in col_ids
    ]

    # Cập nhật column_names_original (tên cột gốc không lowercase)
    ndb["column_names_original"] = [
        [_re_index(index_map=table_ids, ind=t_idx), col_name]
        for t_idx, col_name in db_dict["column_names_original"]
        if t_idx in table_ids or t_idx == -1
    ]

    # Cập nhật primary_keys: chỉ giữ khóa chính của bảng đã lọc
    # Tính lại chỉ số theo col_ids mới
    ndb["primary_keys"] = [
        _re_index(index_map=col_ids, ind=pk)
        for pk in db_dict["primary_keys"]
        if pk in col_ids
    ]

    # Cập nhật foreign_keys: chỉ giữ khóa ngoại trong nội bộ bảng đã lọc
    ndb["foreign_keys"] = [
        [_re_index(index_map=col_ids, ind=from_idx), _re_index(index_map=col_ids, ind=to_idx)]
        for from_idx, to_idx in db_dict["foreign_keys"]
        if from_idx in col_ids and to_idx in col_ids
    ]

    # Cập nhật tên bảng
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
