from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Iterable


class SqliteStorage:
    """Small, process-safe SQLite store for assessment records."""

    def __init__(self, database_path: Path) -> None:
        self.backend = "sqlite"
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()

    @staticmethod
    def _serialize(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return int(value)
        if value is None:
            return ""
        return value

    @staticmethod
    def _identifier(value: str) -> str:
        if not value.replace("_", "").isalnum() or value[0].isdigit():
            raise ValueError(f"Unsafe SQLite identifier: {value!r}")
        return f'"{value}"'

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def append(self, table: str, fieldnames: Iterable[str], record: dict[str, Any]) -> None:
        table_name = table.removesuffix(".csv")
        fields = list(fieldnames)
        quoted_table = self._identifier(table_name)
        quoted_fields = [self._identifier(field) for field in fields]
        values = [self._serialize(record.get(field, "")) for field in fields]

        with self._schema_lock, self._connect() as connection:
            columns = ", ".join(f"{field} TEXT" for field in quoted_fields)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {quoted_table} "
                f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
            )
            existing = {row[1] for row in connection.execute(f"PRAGMA table_info({quoted_table})")}
            for field, quoted_field in zip(fields, quoted_fields):
                if field not in existing:
                    connection.execute(f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_field} TEXT")
            placeholders = ", ".join("?" for _ in fields)
            connection.execute(
                f"INSERT INTO {quoted_table} ({', '.join(quoted_fields)}) VALUES ({placeholders})",
                values,
            )

    def upsert(
        self,
        table: str,
        fieldnames: Iterable[str],
        record: dict[str, Any],
        conflict_fields: Iterable[str],
    ) -> None:
        fields = list(fieldnames)
        conflicts = list(conflict_fields)
        quoted_table = self._identifier(table)
        quoted_fields = [self._identifier(field) for field in fields]
        values = [self._serialize(record.get(field, "")) for field in fields]
        with self._schema_lock, self._connect() as connection:
            columns = ", ".join(f"{field} TEXT" for field in quoted_fields)
            connection.execute(f"CREATE TABLE IF NOT EXISTS {quoted_table} (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})")
            existing = {row[1] for row in connection.execute(f"PRAGMA table_info({quoted_table})")}
            for field, quoted_field in zip(fields, quoted_fields):
                if field not in existing:
                    connection.execute(f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_field} TEXT")
            index_name = self._identifier(f"uq_{table}_{'_'.join(conflicts)}")
            conflict_sql = ", ".join(self._identifier(field) for field in conflicts)
            connection.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {quoted_table} ({conflict_sql})")
            updates = ", ".join(
                f"{field}=excluded.{field}" for field in quoted_fields
                if field not in {self._identifier(item) for item in conflicts}
            )
            placeholders = ", ".join("?" for _ in fields)
            connection.execute(
                f"INSERT INTO {quoted_table} ({', '.join(quoted_fields)}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {updates}",
                values,
            )

    def participant_exists(self, participant_id: str) -> bool:
        with self._connect() as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='participants'"
            ).fetchone()
            if not table:
                return False
            row = connection.execute(
                "SELECT 1 FROM participants WHERE participant_id = ? LIMIT 1",
                (participant_id,),
            ).fetchone()
            return row is not None

    def next_participant_id(self) -> str:
        with self._schema_lock, self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS participant_id_sequence "
                "(name TEXT PRIMARY KEY, current_value INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO participant_id_sequence (name, current_value) VALUES ('participant', 0)"
            )
            connection.execute(
                "UPDATE participant_id_sequence SET current_value = current_value + 1 WHERE name = 'participant'"
            )
            value = connection.execute(
                "SELECT current_value FROM participant_id_sequence WHERE name = 'participant'"
            ).fetchone()[0]
            return f"COG{value:04d}"

    def health_check(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def list_records(self, table: str, limit: int = 500) -> list[dict[str, Any]]:
        table_name = table.removesuffix(".csv")
        quoted_table = self._identifier(table_name)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if not exists:
                return []
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT * FROM {quoted_table} ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]


class PostgresStorage:
    """PostgreSQL implementation with the same append contract used by the API."""

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

    def append(self, table: str, fieldnames: Iterable[str], record: dict[str, Any]) -> None:
        table_name = table.removesuffix(".csv")
        fields = list(fieldnames)
        quoted_table = self._identifier(table_name)
        quoted_fields = [self._identifier(field) for field in fields]
        values = [self._serialize(record.get(field, "")) for field in fields]
        with self._schema_lock, self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                columns = ", ".join(f"{field} TEXT" for field in quoted_fields)
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {quoted_table} "
                    f"(id BIGSERIAL PRIMARY KEY, {columns})"
                )
                for quoted_field in quoted_fields:
                    cursor.execute(f"ALTER TABLE {quoted_table} ADD COLUMN IF NOT EXISTS {quoted_field} TEXT")
                placeholders = ", ".join("%s" for _ in fields)
                cursor.execute(
                    f"INSERT INTO {quoted_table} ({', '.join(quoted_fields)}) VALUES ({placeholders})",
                    values,
                )

    def upsert(
        self,
        table: str,
        fieldnames: Iterable[str],
        record: dict[str, Any],
        conflict_fields: Iterable[str],
    ) -> None:
        fields = list(fieldnames)
        conflicts = list(conflict_fields)
        quoted_table = self._identifier(table)
        quoted_fields = [self._identifier(field) for field in fields]
        values = [self._serialize(record.get(field, "")) for field in fields]
        with self._schema_lock, self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                columns = ", ".join(f"{field} TEXT" for field in quoted_fields)
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {quoted_table} (id BIGSERIAL PRIMARY KEY, {columns})")
                for quoted_field in quoted_fields:
                    cursor.execute(f"ALTER TABLE {quoted_table} ADD COLUMN IF NOT EXISTS {quoted_field} TEXT")
                index_name = self._identifier(f"uq_{table}_{'_'.join(conflicts)}")
                conflict_sql = ", ".join(self._identifier(field) for field in conflicts)
                index_predicate = " WHERE \"record_kind\" = 'summary'" if table == "tracking_data" else ""
                cursor.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {quoted_table} ({conflict_sql}){index_predicate}")
                conflict_set = {self._identifier(item) for item in conflicts}
                updates = ", ".join(
                    f"{field}=EXCLUDED.{field}" for field in quoted_fields if field not in conflict_set
                )
                placeholders = ", ".join("%s" for _ in fields)
                cursor.execute(
                    f"INSERT INTO {quoted_table} ({', '.join(quoted_fields)}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({conflict_sql}){index_predicate} DO UPDATE SET {updates}",
                    values,
                )

    def participant_exists(self, participant_id: str) -> bool:
        with self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass('public.participants')")
                if cursor.fetchone()[0] is None:
                    return False
                cursor.execute(
                    "SELECT 1 FROM participants WHERE participant_id = %s LIMIT 1",
                    (participant_id,),
                )
                return cursor.fetchone() is not None

    def next_participant_id(self) -> str:
        with self._psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE SEQUENCE IF NOT EXISTS participant_id_sequence START 1")
                cursor.execute("SELECT nextval('participant_id_sequence')")
                value = cursor.fetchone()[0]
                return f"COG{value:04d}"

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


def create_storage(database_url: str, fallback_path: Path):
    if database_url:
        return PostgresStorage(database_url)
    return SqliteStorage(fallback_path)
