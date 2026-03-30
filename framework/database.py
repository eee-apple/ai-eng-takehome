"""Database tooling for executing SQL queries against the consolidated DuckDB database.

All CTU Relational databases are consolidated into a single DuckDB file (hecks.duckdb)
with each original database as its own schema. Queries use schema.table syntax.

Example:
    >>> result = execute_query("SELECT * FROM financial.account LIMIT 10")
    >>> if result.is_success:
    ...     print(result.dataframe)
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

# Path to the consolidated database file
DATABASE_PATH = Path(__file__).parent.parent / "hecks.duckdb"


@dataclass
class QueryExecutionResult:
    """Result of executing a SQL query.

    Attributes:
        dataframe: The query results as a Polars DataFrame, or None if error.
        error_message: Error message if execution failed, None otherwise.
    """

    dataframe: pl.DataFrame | None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        """Return True if the query executed successfully."""
        return self.error_message is None

    @property
    def is_empty(self) -> bool:
        """Return True if the query succeeded but returned no rows."""
        return self.is_success and self.dataframe is not None and self.dataframe.is_empty()


def execute_query(query: str) -> QueryExecutionResult:
    """Execute a SQL query against the consolidated database.

    Queries should use schema.table syntax (e.g., "SELECT * FROM financial.account").

    Args:
        query: SQL query string with schema-qualified table names.

    Returns:
        QueryExecutionResult containing either:
        - A Polars DataFrame with the query results (on success)
        - An error message describing what went wrong (on failure)

    Example:
        >>> result = execute_query("SELECT * FROM financial.account LIMIT 10")
        >>> if result.is_success:
        ...     print(result.dataframe)
        ... else:
        ...     print(f"Error: {result.error_message}")
    """
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        result = conn.execute(query)
        df = pl.DataFrame(result.fetch_arrow_table())
        return QueryExecutionResult(dataframe=df)
    except duckdb.Error as e:
        return QueryExecutionResult(dataframe=None, error_message=f"DuckDB error: {e}")
    except Exception as e:
        return QueryExecutionResult(dataframe=None, error_message=str(e))
    finally:
        if conn is not None:
            conn.close()
