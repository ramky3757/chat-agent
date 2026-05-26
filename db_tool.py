"""
Database tool — read-only wrapper exposed to the Hermes agent.

Supports MySQL and PostgreSQL. Set `db_type: mysql` or `db_type: postgres`
in config.yaml to switch. Write operations are blocked at two layers:
  1. Regex pattern check before the query reaches the database
  2. DB user should also be granted SELECT-only privileges (recommended)
"""

import re
from abc import ABC, abstractmethod
from typing import Any

# ── Safety checks (shared by both backends) ────────────────────────────────

_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|MERGE|UPSERT"
    r"|DROP|TRUNCATE|ALTER|CREATE|RENAME"
    r"|GRANT|REVOKE|LOCK|UNLOCK"
    r"|CALL|EXEC|EXECUTE|LOAD\s+DATA)\b",
    re.IGNORECASE,
)

_ALLOWED_START = re.compile(
    r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN|WITH)\b",
    re.IGNORECASE,
)


def _safe_query(sql: str) -> tuple[bool, str]:
    if _BLOCKED.search(sql):
        return False, (
            "Blocked: only SELECT / SHOW / DESCRIBE / EXPLAIN queries are permitted."
        )
    if not _ALLOWED_START.match(sql):
        return False, (
            "Blocked: query must start with SELECT, SHOW, DESCRIBE, EXPLAIN, or WITH."
        )
    return True, ""


# ── Base class ─────────────────────────────────────────────────────────────

class BaseDBTool(ABC):
    def __init__(self, config: dict) -> None:
        self._cfg = config["database"]
        self.max_rows: int = int(self._cfg.get("max_rows", 200))

    @abstractmethod
    def _connect_with_dict_cursor(self):
        """Return (conn, cursor) where cursor yields dict rows."""
        ...

    @abstractmethod
    def list_tables(self) -> dict[str, Any]: ...

    @abstractmethod
    def describe_table(self, table_name: str) -> dict[str, Any]: ...

    def execute_query(self, query: str) -> dict[str, Any]:
        """Run a read-only SQL query and return structured results."""
        ok, err = _safe_query(query)
        if not ok:
            return {"error": err, "columns": [], "rows": [], "row_count": 0}
        try:
            conn, cursor = self._connect_with_dict_cursor()
            cursor.execute(query)
            rows = [dict(r) for r in cursor.fetchmany(self.max_rows)]
            columns = [d[0] for d in cursor.description] if cursor.description else []
            truncated = len(rows) == self.max_rows
            cursor.close()
            conn.close()
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
                "note": f"Results capped at {self.max_rows} rows." if truncated else "",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "rows": [], "row_count": 0}

    def get_schema(self) -> dict[str, Any]:
        """Full schema — all tables and their columns. Used by /schema command."""
        result = self.list_tables()
        if "error" in result:
            return {"error": result["error"], "tables": [], "schema": {}}
        schema: dict[str, list] = {}
        for table in result["tables"]:
            desc = self.describe_table(table)
            schema[table] = desc.get("columns", [])
        return {"tables": result["tables"], "schema": schema}


# ── MySQL ──────────────────────────────────────────────────────────────────

class MySQLTool(BaseDBTool):
    def _connect_with_dict_cursor(self):
        import mysql.connector
        conn = mysql.connector.connect(
            host=self._cfg["host"],
            port=int(self._cfg.get("port", 3306)),
            user=self._cfg["username"],
            password=self._cfg["password"],
            database=self._cfg["database"],
            connection_timeout=10,
        )
        return conn, conn.cursor(dictionary=True)

    def list_tables(self) -> dict[str, Any]:
        try:
            conn, cursor = self._connect_with_dict_cursor()
            cursor.execute("SHOW TABLES")
            tables = [list(row.values())[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return {"tables": tables, "count": len(tables)}
        except Exception as exc:
            return {"error": str(exc), "tables": []}

    def describe_table(self, table_name: str) -> dict[str, Any]:
        if not re.match(r"^\w+$", table_name):
            return {"error": "Invalid table name."}
        try:
            conn, cursor = self._connect_with_dict_cursor()
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return {"table": table_name, "columns": columns}
        except Exception as exc:
            return {"error": str(exc), "columns": []}


# ── PostgreSQL ─────────────────────────────────────────────────────────────

class PostgreSQLTool(BaseDBTool):
    def _connect_with_dict_cursor(self):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise RuntimeError(
                "psycopg2 is not installed. Run: pip install psycopg2-binary"
            )
        conn = psycopg2.connect(
            host=self._cfg["host"],
            port=int(self._cfg.get("port", 5432)),
            user=self._cfg["username"],
            password=self._cfg["password"],
            dbname=self._cfg["database"],
            connect_timeout=10,
        )
        conn.autocommit = True
        return conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def list_tables(self) -> dict[str, Any]:
        try:
            conn, cursor = self._connect_with_dict_cursor()
            cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' ORDER BY tablename"
            )
            tables = [row["tablename"] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return {"tables": tables, "count": len(tables)}
        except Exception as exc:
            return {"error": str(exc), "tables": []}

    def describe_table(self, table_name: str) -> dict[str, Any]:
        if not re.match(r"^\w+$", table_name):
            return {"error": "Invalid table name."}
        try:
            conn, cursor = self._connect_with_dict_cursor()
            cursor.execute(
                """
                SELECT
                    c.column_name                                           AS "Field",
                    c.data_type                                             AS "Type",
                    c.is_nullable                                           AS "Null",
                    c.column_default                                        AS "Default",
                    CASE WHEN pk.column_name IS NOT NULL THEN 'PRI' ELSE '' END AS "Key"
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                       AND tc.table_schema    = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_name      = %s
                      AND tc.table_schema    = 'public'
                ) pk ON c.column_name = pk.column_name
                WHERE c.table_name   = %s
                  AND c.table_schema = 'public'
                ORDER BY c.ordinal_position
                """,
                (table_name, table_name),
            )
            columns = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return {"table": table_name, "columns": columns}
        except Exception as exc:
            return {"error": str(exc), "columns": []}


# ── Factory ────────────────────────────────────────────────────────────────

def create_db_tool(config: dict) -> BaseDBTool:
    """Return the right DB tool based on config.yaml db_type field."""
    db_type = config["database"].get("db_type", "mysql").lower()
    if db_type in ("postgres", "postgresql"):
        return PostgreSQLTool(config)
    if db_type == "mysql":
        return MySQLTool(config)
    raise ValueError(f"Unsupported db_type '{db_type}'. Use 'mysql' or 'postgres'.")
