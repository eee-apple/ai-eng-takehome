"""Database exploration tools for the SQL agent."""

import duckdb

from framework.agent import Tool
from framework.database import DATABASE_PATH, execute_query

# Pre-compute a schema index at import time (single DB connection) so
# search_tables is fast and doesn't open 600+ connections.
_SCHEMA_INDEX: dict[str, dict[str, list[str]]] = {}  # schema -> table -> [col descriptions]


def _build_index() -> dict[str, dict[str, list[str]]]:
    """Build a full index of schema.table.columns from information_schema."""
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        rows = conn.execute("""
            SELECT table_schema, table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name, ordinal_position
        """).fetchall()

        index: dict[str, dict[str, list[str]]] = {}
        for schema, table, col_name, data_type, is_nullable in rows:
            nullable_str = ", nullable" if is_nullable == "YES" else ""
            desc = f"{col_name} ({data_type}{nullable_str})"
            index.setdefault(schema, {}).setdefault(table, []).append(desc)
        return index
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()


_SCHEMA_INDEX = _build_index()


def _describe_column(schema_name: str, table_name: str, column_name: str) -> str:
    """Get statistics for a column: min, max, count distinct, null count, top values."""
    # Resolve schema/table case-insensitively
    schema_tables = _SCHEMA_INDEX.get(schema_name)
    if not schema_tables:
        for key in _SCHEMA_INDEX:
            if key.lower() == schema_name.lower():
                schema_tables = _SCHEMA_INDEX[key]
                schema_name = key
                break
    if not schema_tables:
        return f"Schema '{schema_name}' not found."
    if table_name not in schema_tables:
        for t in schema_tables:
            if t.lower() == table_name.lower():
                table_name = t
                break

    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT("{column_name}") AS non_null_count,
            COUNT(*) - COUNT("{column_name}") AS null_count,
            COUNT(DISTINCT "{column_name}") AS distinct_count,
            MIN("{column_name}") AS min_value,
            MAX("{column_name}") AS max_value
        FROM "{schema_name}"."{table_name}"
    """
    result = execute_query(query)
    if not result.is_success:
        return f"Error: {result.error_message}"
    df = result.dataframe
    if df is None or df.is_empty():
        return f"No data in {schema_name}.{table_name}.{column_name}"

    row = df.row(0, named=True)
    lines = [
        f"Column stats for {schema_name}.{table_name}.{column_name}:",
        f"  Total rows:     {row['total_rows']}",
        f"  Non-null:       {row['non_null_count']}",
        f"  Null count:     {row['null_count']}",
        f"  Distinct count: {row['distinct_count']}",
        f"  Min:            {row['min_value']}",
        f"  Max:            {row['max_value']}",
    ]

    # Also grab top 10 most frequent values for low-cardinality columns
    if row["distinct_count"] is not None and row["distinct_count"] <= 50:
        freq_query = f"""
            SELECT "{column_name}" AS value, COUNT(*) AS cnt
            FROM "{schema_name}"."{table_name}"
            WHERE "{column_name}" IS NOT NULL
            GROUP BY "{column_name}"
            ORDER BY cnt DESC
            LIMIT 10
        """
        freq_result = execute_query(freq_query)
        if freq_result.is_success and freq_result.dataframe is not None:
            fdf = freq_result.dataframe
            top = [(str(r["value"]), r["cnt"]) for r in fdf.iter_rows(named=True)]
            lines.append("  Top values:     " + ", ".join(f"{v} ({c})" for v, c in top))

    return "\n".join(lines)


def _sample_values(schema_name: str, table_name: str, column_name: str) -> str:
    """Get distinct sample values from a column to understand its content."""
    result = execute_query(
        f'SELECT DISTINCT "{column_name}" FROM "{schema_name}"."{table_name}" '
        f'WHERE "{column_name}" IS NOT NULL LIMIT 20'
    )
    if not result.is_success:
        return f"Error: {result.error_message}"
    df = result.dataframe
    if df is None or df.is_empty():
        return f"No non-null values in {schema_name}.{table_name}.{column_name}"
    values = df[column_name].to_list()
    return f"Sample values for {schema_name}.{table_name}.{column_name} ({df.height} shown):\n  {values}"


def _execute_sql(query: str) -> str:
    result = execute_query(query)
    if not result.is_success:
        return f"Query error: {result.error_message}"
    df = result.dataframe
    if df is None:
        return "Query returned no dataframe."
    if df.is_empty():
        return f"Query returned 0 rows. Columns: {df.columns}"
    # Limit output to avoid overwhelming the context window
    max_rows = 15
    if df.height > max_rows:
        preview = df.head(max_rows)
        return f"Query returned {df.height} rows, {df.width} columns. Showing first {max_rows}:\n{preview}"
    return f"Query returned {df.height} rows, {df.width} columns:\n{df}"


DESCRIBE_COLUMN = Tool(
    name="describe_column",
    description=(
        "Get column statistics: row count, nulls, distinct count, min/max, "
        "and top frequent values for low-cardinality columns."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema_name": {"type": "string", "description": "The schema name."},
            "table_name": {"type": "string", "description": "The table name."},
            "column_name": {"type": "string", "description": "The column to describe."},
        },
        "required": ["schema_name", "table_name", "column_name"],
    },
    function=_describe_column,
)

SAMPLE_VALUES = Tool(
    name="sample_values",
    description=(
        "Get up to 20 distinct sample values from a column. "
        "Useful for understanding codes, categories, or ID formats."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema_name": {
                "type": "string",
                "description": "The schema name.",
            },
            "table_name": {
                "type": "string",
                "description": "The table name.",
            },
            "column_name": {
                "type": "string",
                "description": "The column name to sample values from.",
            },
        },
        "required": ["schema_name", "table_name", "column_name"],
    },
    function=_sample_values,
)

EXECUTE_SQL = Tool(
    name="execute_sql",
    description=(
        "Execute a SQL query and return results. "
        "Use schema.table syntax (e.g., 'SELECT * FROM schema.table LIMIT 5')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL query to execute. Always use schema.table syntax.",
            },
        },
        "required": ["query"],
    },
    function=_execute_sql,
)
