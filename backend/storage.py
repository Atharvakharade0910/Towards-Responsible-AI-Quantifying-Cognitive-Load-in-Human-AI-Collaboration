from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Iterable


class PostgresStorage:
    """Process-safe PostgreSQL store for assessment records."""

    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL") from exc
        self.backend = "postgresql"
        self.database_url = database_url
        self._psycopg = psycopg
        self._schema_lock = threading.Lock()

    @staticmethod
    def _serialize(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return str(value).lower()
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _identifier(value: str) -> str:
        if not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError(f"Unsafe PostgreSQL identifier: {value!r}")
        return f'"{value}"'

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        with self._psycopg.connect(self.database_url) as connection:
            yield connection

    def _ensure_table(self, cursor: Any, table: str, fields: list[str]) -> None:
        quoted_table = self._identifier(table)
        quoted_fields = [self._identifier(field) for field in fields]
        columns = ", ".join(f"{field} TEXT" for field in quoted_fields)
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {quoted_table} "
            f"(id BIGSERIAL PRIMARY KEY, {columns})"
        )
        for field in quoted_fields:
            cursor.execute(f"ALTER TABLE {quoted_table} ADD COLUMN IF NOT EXISTS {field} TEXT")

    def append(self, table: str, fieldnames: Iterable[str], record: dict[str, Any]) -> None:
        table_name = table.removesuffix(".csv")
        fields = list(fieldnames)
        values = [self._serialize(record.get(field, "")) for field in fields]
        quoted_table = self._identifier(table_name)
        quoted_fields = [self._identifier(field) for field in fields]
        with self._schema_lock, self._connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_table(cursor, table_name, fields)
                placeholders = ", ".join("%s" for _ in fields)
                cursor.execute(
                    f"INSERT INTO {quoted_table} ({', '.join(quoted_fields)}) VALUES ({placeholders})",
                    values,
                )

    def ensure_table(self, table: str, fields: Iterable[str]) -> None:
        """Create a PostgreSQL table and any missing columns without inserting data."""
        table_name = table.removesuffix(".csv")
        with self._schema_lock, self._connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_table(cursor, table_name, list(fields))

    def add_column(self, table: str, field: str, sql_type: str = "TEXT") -> None:
        """Add one validated application-defined column to a table."""
        table_name = table.removesuffix(".csv")
        quoted_table = self._identifier(table_name)
        quoted_field = self._identifier(field)
        allowed_types = {"TEXT", "DOUBLE PRECISION", "BOOLEAN", "TIMESTAMPTZ"}
        normalized_type = sql_type.strip().upper()
        if normalized_type not in allowed_types:
            raise ValueError(f"Unsupported PostgreSQL column type: {sql_type!r}")
        with self._schema_lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"ALTER TABLE {quoted_table} ADD COLUMN IF NOT EXISTS "
                    f"{quoted_field} {normalized_type}"
                )

    def update_where(
        self,
        table: str,
        values: dict[str, Any],
        conditions: dict[str, Any],
    ) -> int:
        """Update rows using validated identifiers and equality predicates."""
        if not values or not conditions:
            return 0
        table_name = table.removesuffix(".csv")
        quoted_table = self._identifier(table_name)
        set_fields = list(values)
        where_fields = list(conditions)
        quoted_set = [self._identifier(field) for field in set_fields]
        quoted_where = [self._identifier(field) for field in where_fields]
        with self._schema_lock, self._connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_table(cursor, table_name, [*set_fields, *where_fields])
                assignments = ", ".join(f"{field} = %s" for field in quoted_set)
                predicates = " AND ".join(f"{field} = %s" for field in quoted_where)
                cursor.execute(
                    f"UPDATE {quoted_table} SET {assignments} WHERE {predicates}",
                    [self._serialize(values[field]) for field in set_fields]
                    + [self._serialize(conditions[field]) for field in where_fields],
                )
                return cursor.rowcount

    def table_columns(self, table: str) -> list[str]:
        table_name = table.removesuffix(".csv")
        with self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (table_name,),
                )
                return [row[0] for row in cursor.fetchall()]

    def upsert(
        self,
        table: str,
        fieldnames: Iterable[str],
        record: dict[str, Any],
        conflict_fields: Iterable[str],
    ) -> None:
        table_name = table.removesuffix(".csv")
        fields = list(fieldnames)
        conflicts = list(conflict_fields)
        values = [self._serialize(record.get(field, "")) for field in fields]
        quoted_table = self._identifier(table_name)
        quoted_fields = [self._identifier(field) for field in fields]
        conflict_sql = ", ".join(self._identifier(field) for field in conflicts)
        index_name = self._identifier(f"uq_{table_name}_{'_'.join(conflicts)}")
        index_predicate = " WHERE \"record_kind\" = 'summary'" if table_name == "tracking_data" else ""
        with self._schema_lock, self._connect() as connection:
            with connection.cursor() as cursor:
                self._ensure_table(cursor, table_name, fields)
                cursor.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} "
                    f"ON {quoted_table} ({conflict_sql}){index_predicate}"
                )
                placeholders = ", ".join("%s" for _ in fields)
                conflict_set = {self._identifier(field) for field in conflicts}
                updates = ", ".join(
                    f"{field}=EXCLUDED.{field}"
                    for field in quoted_fields
                    if field not in conflict_set
                )
                cursor.execute(
                    f"INSERT INTO {quoted_table} ({', '.join(quoted_fields)}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({conflict_sql}){index_predicate} DO UPDATE SET {updates}",
                    values,
                )

    def participant_exists(self, participant_id: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass('public.participants')")
                if cursor.fetchone()[0] is None:
                    return False
                cursor.execute("SELECT 1 FROM participants WHERE participant_id = %s LIMIT 1", (participant_id,))
                return cursor.fetchone() is not None

    def next_participant_id(self) -> str:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                # Serialize allocation and reset a previously-used empty
                # database so the first real participant always starts at
                # COG0001 without creating duplicate IDs under concurrency.
                cursor.execute("SELECT pg_advisory_xact_lock(hashtext('cognitrack_participant_ids'))")
                cursor.execute("CREATE SEQUENCE IF NOT EXISTS participant_id_sequence START 1")
                cursor.execute("SELECT to_regclass('public.participants')")
                participants_table = cursor.fetchone()[0]
                if participants_table is None:
                    cursor.execute("SELECT setval('participant_id_sequence', 1, false)")
                else:
                    cursor.execute("SELECT COUNT(*) FROM participants")
                    if cursor.fetchone()[0] == 0:
                        cursor.execute("SELECT setval('participant_id_sequence', 1, false)")
                cursor.execute("SELECT nextval('participant_id_sequence')")
                return f"COG{cursor.fetchone()[0]:04d}"

    def health_check(self) -> bool:
        with self._psycopg.connect(self.database_url, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0] == 1

    def list_records(self, table: str, limit: int = 500) -> list[dict[str, Any]]:
        table_name = table.removesuffix(".csv")
        quoted_table = self._identifier(table_name)
        from psycopg.rows import dict_row
        with self._psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
                if cursor.fetchone()["to_regclass"] is None:
                    return []
                cursor.execute(f"SELECT * FROM {quoted_table} ORDER BY id DESC LIMIT %s", (limit,))
                return list(cursor.fetchall())

    def count_records(self, table: str) -> int:
        """Return the row count for an application table, or zero if absent."""
        table_name = table.removesuffix(".csv")
        quoted_table = self._identifier(table_name)
        with self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
                if cursor.fetchone()[0] is None:
                    return 0
                cursor.execute(f"SELECT COUNT(*) FROM {quoted_table}")
                return int(cursor.fetchone()[0])

    def list_records_page(
        self,
        table: str,
        limit: int = 50,
        offset: int = 0,
        component_type: str | None = None,
        record_kind: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return one bounded page and total count for Admin data views."""
        table_name = table.removesuffix(".csv")
        quoted_table = self._identifier(table_name)
        from psycopg.rows import dict_row
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        with self._psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
                if cursor.fetchone()["to_regclass"] is None:
                    return [], 0
                where = ""
                params: list[Any] = []
                if component_type is not None:
                    where = " WHERE \"component_type\" = %s"
                    params.append(component_type)
                if record_kind is not None:
                    where += " AND \"record_kind\" = %s" if where else " WHERE \"record_kind\" = %s"
                    params.append(record_kind)
                cursor.execute(f"SELECT COUNT(*) AS total FROM {quoted_table}{where}", params)
                total = int(cursor.fetchone()["total"])
                cursor.execute(
                    f"SELECT * FROM {quoted_table}{where} ORDER BY id DESC LIMIT %s OFFSET %s",
                    [*params, safe_limit, safe_offset],
                )
                return list(cursor.fetchall()), total


class SqliteStorage:
    """SQLite storage used by the isolated API test suite and local fallback tools."""

    backend = "sqlite"

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._schema_lock = threading.Lock()

    @staticmethod
    def _serialize(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return int(value)
        return value

    @staticmethod
    def _identifier(value: str) -> str:
        if not value or not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError(f"Unsafe SQLite identifier: {value!r}")
        return f'"{value}"'

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _ensure_table(self, connection: sqlite3.Connection, table: str, fields: list[str]) -> None:
        quoted_table = self._identifier(table)
        quoted_fields = [self._identifier(field) for field in fields]
        columns = ", ".join(f"{field} TEXT" for field in quoted_fields)
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {quoted_table} (id INTEGER PRIMARY KEY AUTOINCREMENT"
            f"{', ' + columns if columns else ''})"
        )
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({quoted_table})")}
        for field in fields:
            if field not in existing:
                connection.execute(f"ALTER TABLE {quoted_table} ADD COLUMN {self._identifier(field)} TEXT")

    def append(self, table: str, fieldnames: Iterable[str], record: dict[str, Any]) -> None:
        table_name = table.removesuffix(".csv")
        fields = list(fieldnames)
        with self._schema_lock, self._connect() as connection:
            self._ensure_table(connection, table_name, fields)
            quoted = ", ".join(self._identifier(field) for field in fields)
            placeholders = ", ".join("?" for _ in fields)
            connection.execute(
                f"INSERT INTO {self._identifier(table_name)} ({quoted}) VALUES ({placeholders})",
                [self._serialize(record.get(field, "")) for field in fields],
            )

    def ensure_table(self, table: str, fields: Iterable[str]) -> None:
        table_name = table.removesuffix(".csv")
        with self._schema_lock, self._connect() as connection:
            self._ensure_table(connection, table_name, list(fields))

    def add_column(self, table: str, field: str, sql_type: str = "TEXT") -> None:
        table_name = table.removesuffix(".csv")
        with self._schema_lock, self._connect() as connection:
            self._ensure_table(connection, table_name, [])
            existing = {row[1] for row in connection.execute(f"PRAGMA table_info({self._identifier(table_name)})")}
            if field not in existing:
                connection.execute(f"ALTER TABLE {self._identifier(table_name)} ADD COLUMN {self._identifier(field)} TEXT")

    def update_where(self, table: str, values: dict[str, Any], conditions: dict[str, Any]) -> int:
        if not values or not conditions:
            return 0
        table_name = table.removesuffix(".csv")
        with self._schema_lock, self._connect() as connection:
            self._ensure_table(connection, table_name, [*values, *conditions])
            assignments = ", ".join(f"{self._identifier(field)} = ?" for field in values)
            predicates = " AND ".join(f"{self._identifier(field)} = ?" for field in conditions)
            result = connection.execute(
                f"UPDATE {self._identifier(table_name)} SET {assignments} WHERE {predicates}",
                [self._serialize(values[field]) for field in values] + [self._serialize(conditions[field]) for field in conditions],
            )
            return result.rowcount

    def table_columns(self, table: str) -> list[str]:
        table_name = table.removesuffix(".csv")
        with self._connect() as connection:
            return [row[1] for row in connection.execute(f"PRAGMA table_info({self._identifier(table_name)})")]

    def upsert(self, table: str, fieldnames: Iterable[str], record: dict[str, Any], conflict_fields: Iterable[str]) -> None:
        table_name = table.removesuffix(".csv")
        fields = list(fieldnames)
        conflicts = list(conflict_fields)
        with self._schema_lock, self._connect() as connection:
            self._ensure_table(connection, table_name, fields)
            conflict_sql = ", ".join(self._identifier(field) for field in conflicts)
            index_name = self._identifier(f"uq_{table_name}_{'_'.join(conflicts)}")
            connection.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {self._identifier(table_name)} ({conflict_sql})"
            )
            quoted_fields = ", ".join(self._identifier(field) for field in fields)
            placeholders = ", ".join("?" for _ in fields)
            updates = ", ".join(
                f"{self._identifier(field)} = excluded.{self._identifier(field)}"
                for field in fields if field not in conflicts
            )
            connection.execute(
                f"INSERT INTO {self._identifier(table_name)} ({quoted_fields}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {updates}",
                [self._serialize(record.get(field, "")) for field in fields],
            )

    def participant_exists(self, participant_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM participants WHERE participant_id = ? LIMIT 1", (participant_id,)
            ).fetchone()
            return row is not None

    def next_participant_id(self) -> str:
        with self._schema_lock, self._connect() as connection:
            self._ensure_table(connection, "participants", ["participant_id"])
            row = connection.execute(
                "SELECT participant_id FROM participants WHERE participant_id LIKE 'COG%'"
            ).fetchall()
            numbers = [int(str(item[0])[3:]) for item in row if str(item[0])[3:].isdigit()]
            return f"COG{(max(numbers, default=0) + 1):04d}"

    def health_check(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def list_records(self, table: str, limit: int = 500) -> list[dict[str, Any]]:
        table_name = table.removesuffix(".csv")
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
            ).fetchone()
            if exists is None:
                return []
            rows = connection.execute(
                f"SELECT * FROM {self._identifier(table_name)} ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            columns = [description[0] for description in connection.execute(
                f"SELECT * FROM {self._identifier(table_name)} LIMIT 0"
            ).description]
            return [dict(zip(columns, row)) for row in rows]

    def count_records(self, table: str) -> int:
        """Return the row count for an application table, or zero if absent."""
        table_name = table.removesuffix(".csv")
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
            ).fetchone()
            if exists is None:
                return 0
            return int(connection.execute(
                f"SELECT COUNT(*) FROM {self._identifier(table_name)}"
            ).fetchone()[0])

    def list_records_page(self, table: str, limit: int = 50, offset: int = 0, component_type: str | None = None, record_kind: str | None = None) -> tuple[list[dict[str, Any]], int]:
        records = self.list_records(table, 100000)
        if component_type is not None:
            records = [row for row in records if row.get("component_type") == component_type]
        if record_kind is not None:
            records = [row for row in records if row.get("record_kind") == record_kind]
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        return records[safe_offset:safe_offset + safe_limit], len(records)


def create_storage(database_url: str) -> PostgresStorage:
    if not database_url:
        raise RuntimeError("DATABASE_URL is required. Configure PostgreSQL before starting the app.")
    return PostgresStorage(database_url)
