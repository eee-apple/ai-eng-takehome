"""Schema discovery tools for structured database exploration.

Supports a 3-step workflow:
1. list_all_schemas — scan all schemas to pick the right one
2. get_full_schema — see all tables + columns in chosen schema
3. sample_table — see actual data from specific tables
"""

from pathlib import Path

import duckdb

from framework.agent import Tool
from framework.database import DATABASE_PATH
from tools.database_tools import _SCHEMA_INDEX

# ── Schema → business rules guide mapping ──────────────────────────────────────

GUIDES_DIR = Path(__file__).parent.parent / "evaluation" / "data" / "guides"


def _find_guide_for_schema(schema_name: str) -> Path | None:
    """Find a business rules guide file for a schema.

    Searches guide file contents for the schema name.
    """
    if not GUIDES_DIR.exists():
        return None
    for g in GUIDES_DIR.glob("*.md"):
        content = g.read_text(errors="ignore")
        if schema_name in content:
            return g
    return None

# ── Pre-compute row counts at import time ─────────────────────────────────────

_ROW_COUNTS: dict[str, int] = {}  # "schema.table" -> row count


def _build_row_counts() -> dict[str, int]:
    """Get row counts for all tables in a single connection."""
    counts: dict[str, int] = {}
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        for schema, tables in _SCHEMA_INDEX.items():
            for table in tables:
                try:
                    rc = conn.execute(
                        f'SELECT COUNT(*) FROM "{schema}"."{table}"'
                    ).fetchone()[0]
                    counts[f"{schema}.{table}"] = rc
                except Exception:
                    counts[f"{schema}.{table}"] = -1
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()
    return counts


_ROW_COUNTS = _build_row_counts()

# ── Tool 1: List all schemas ─────────────────────────────────────────────────


def _list_all_schemas() -> str:
    """List every schema with its tables and row counts.

    Returns a compact directory the agent can scan to pick the right schema.
    """
    parts: list[str] = []
    parts.append(f"Database has {len(_SCHEMA_INDEX)} schemas, {sum(len(t) for t in _SCHEMA_INDEX.values())} tables total.\n")

    for schema in sorted(_SCHEMA_INDEX.keys()):
        tables = _SCHEMA_INDEX[schema]
        # Build compact table list with row counts
        table_entries: list[str] = []
        for table in sorted(tables.keys()):
            rc = _ROW_COUNTS.get(f"{schema}.{table}", -1)
            if rc >= 0:
                table_entries.append(f"{table}({rc:,})")
            else:
                table_entries.append(table)

        parts.append(f"{schema} [{len(tables)} tables]: {', '.join(table_entries)}")

    return "\n".join(parts)


LIST_ALL_SCHEMAS = Tool(
    name="list_all_schemas",
    description=(
        "List all schemas with table names and row counts. "
        "Call this first to pick the right schema."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    function=_list_all_schemas,
)


# ── Tool 2: Get full schema ──────────────────────────────────────────────────


def _get_full_schema(schema_name: str) -> str:
    """Get all tables and columns for a schema, with row counts."""
    if schema_name not in _SCHEMA_INDEX:
        # Try case-insensitive match
        for s in _SCHEMA_INDEX:
            if s.lower() == schema_name.lower():
                schema_name = s
                break
        else:
            return f"Schema '{schema_name}' not found. Use list_all_schemas to see available schemas."

    tables = _SCHEMA_INDEX[schema_name]
    parts: list[str] = []
    parts.append(f"## Schema: {schema_name} ({len(tables)} tables)\n")

    for table in sorted(tables.keys()):
        columns = tables[table]
        rc = _ROW_COUNTS.get(f"{schema_name}.{table}", -1)
        rc_str = f"{rc:,} rows" if rc >= 0 else "unknown rows"

        parts.append(f"### {schema_name}.{table} ({rc_str})")
        for col in columns:
            parts.append(f"  - {col}")
        parts.append("")

    # Auto-append business rules if available for this schema
    guide_path = _find_guide_for_schema(schema_name)
    if guide_path and guide_path.exists():
        rules_content = guide_path.read_text(errors="ignore")
        parts.append("\n---\n")
        parts.append("## ⚠️ BUSINESS RULES (MUST READ AND APPLY)\n")
        parts.append(rules_content)

    return "\n".join(parts)


GET_FULL_SCHEMA = Tool(
    name="get_full_schema",
    description=(
        "Get all tables, columns, and business rules for a schema. "
        "Call after list_all_schemas to get full details."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema_name": {
                "type": "string",
                "description": "The schema name to explore (e.g., 'financial', 'Credit', 'Airline').",
            },
        },
        "required": ["schema_name"],
    },
    function=_get_full_schema,
)


# ── Tool 3: Sample table ─────────────────────────────────────────────────────


def _sample_table(schema_name: str, table_name: str, num_rows: int = 5) -> str:
    """Show sample rows from a table to understand the actual data."""
    num_rows = min(num_rows, 10)  # Cap at 10

    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        result = conn.execute(
            f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT {num_rows}'
        ).fetchdf()

        if result.empty:
            return f"{schema_name}.{table_name} is empty."

        # Format as readable text
        parts: list[str] = []
        parts.append(f"## Sample from {schema_name}.{table_name} ({num_rows} rows)\n")

        # Column headers
        cols = list(result.columns)
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("| " + " | ".join(["---"] * len(cols)) + " |")

        # Data rows
        for _, row in result.iterrows():
            vals = []
            for col in cols:
                v = str(row[col])
                if len(v) > 40:
                    v = v[:37] + "..."
                vals.append(v)
            parts.append("| " + " | ".join(vals) + " |")

        return "\n".join(parts)
    except Exception as e:
        return f"Error sampling {schema_name}.{table_name}: {e}"
    finally:
        if conn is not None:
            conn.close()


def _summarize_table(schema_name: str, table_name: str) -> str:
    """Run DuckDB's SUMMARIZE on a table to get column-level statistics."""
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        df = conn.execute(
            f'SUMMARIZE "{schema_name}"."{table_name}"'
        ).fetchdf()
        if df.empty:
            return f"No data in {schema_name}.{table_name}."

        lines: list[str] = []
        lines.append(f"## SUMMARIZE {schema_name}.{table_name}")
        lines.append("| Column | Type | Min | Max | Distinct | Null% |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in df.iterrows():
            col_name = str(row.get("column_name", ""))
            col_type = str(row.get("column_type", ""))
            min_val = str(row.get("min", ""))[:40]
            max_val = str(row.get("max", ""))[:40]
            distinct = int(row.get("approx_unique", 0))
            null_pct = row.get("null_percentage", 0)
            lines.append(
                f"| {col_name} | {col_type} | {min_val} "
                f"| {max_val} | {distinct} "
                f"| {null_pct:.0f}% |"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error summarizing {schema_name}.{table_name}: {e}"
    finally:
        if conn is not None:
            conn.close()


SUMMARIZE_TABLE = Tool(
    name="summarize_table",
    description=(
        "Get column-level statistics: type, min/max, distinct count, null percentage. "
        "More informative than sample_table for understanding distributions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema_name": {
                "type": "string",
                "description": "The schema name (e.g., 'financial').",
            },
            "table_name": {
                "type": "string",
                "description": "The table name (e.g., 'loan').",
            },
        },
        "required": ["schema_name", "table_name"],
    },
    function=_summarize_table,
)


SAMPLE_TABLE = Tool(
    name="sample_table",
    description="Show sample rows from a table to verify what the data looks like.",
    parameters={
        "type": "object",
        "properties": {
            "schema_name": {
                "type": "string",
                "description": "The schema name (e.g., 'financial').",
            },
            "table_name": {
                "type": "string",
                "description": "The table name (e.g., 'loan').",
            },
            "num_rows": {
                "type": "integer",
                "description": "Number of sample rows (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["schema_name", "table_name"],
    },
    function=_sample_table,
)
