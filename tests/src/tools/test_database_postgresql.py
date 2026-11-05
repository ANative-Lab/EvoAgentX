import os
from unittest.mock import MagicMock

import pytest

try:
    import psycopg2
    import psycopg2.sql

    HAS_PSYCOPG2 = True
except ImportError:  # pragma: no cover
    HAS_PSYCOPG2 = False

from evoagentx.tools.database_postgresql import PostgreSQLDatabase

PAYLOAD = "information_schema.tables WHERE table_name='users'--"


def _make_db(collection_name="public"):
    """Build a PostgreSQLDatabase whose remote init is fully mocked."""
    db = PostgreSQLDatabase.__new__(PostgreSQLDatabase)
    db.conn = MagicMock()
    db.file_based_mode = False
    db._is_initialized = True
    db.is_local_database = False
    db.local_path = None
    db.auto_save = False
    db.database_name = collection_name
    return db


def _executed_statements(db):
    """Return every first positional arg passed to cursor.execute."""
    cursor = db.conn.cursor.return_value.__enter__.return_value
    return [call.args[0] for call in cursor.execute.call_args_list]


@pytest.mark.skipif(not HAS_PSYCOPG2, reason="psycopg2 not installed")
class TestGetCollectionInfoSqlInjection:
    def test_identifier_is_parameterized_not_interpolated(self):
        """Regression test for issue #287: the table name must reach the
        driver as a psycopg2.sql.Identifier, never interpolated into a
        plain SQL string."""
        db = _make_db()
        db.get_collection_info(PAYLOAD)

        statements = _executed_statements(db)
        assert statements, "expected a statement to be executed"
        for stmt in statements:
            # No raw interpolated SQL may contain the payload verbatim.
            assert not (isinstance(stmt, str) and PAYLOAD in stmt)
        count_stmt = statements[0]
        assert isinstance(count_stmt, psycopg2.sql.Composed)
        # The payload is carried as a quoted identifier inside the Composed.
        assert any(
            isinstance(part, psycopg2.sql.Identifier)
            and getattr(part, "strings", None) == (PAYLOAD,)
            for part in count_stmt.seq
        )

    def test_injection_is_never_executed_as_sql(self):
        """The whole payload is one quoted identifier: it cannot split the
        statement, so the injected WHERE clause never runs."""
        db = _make_db()
        db.get_collection_info(PAYLOAD)

        count_stmt = _executed_statements(db)[0]
        literal = "".join(
            part.string for part in count_stmt.seq if isinstance(part, psycopg2.sql.SQL)
        )
        # The literal SQL offers no surface: the payload only rides along as
        # the quoted identifier, never as executable statement text.
        assert literal == "SELECT COUNT(*) as row_count FROM "
        assert PAYLOAD not in literal

    def test_rollback_on_error_keeps_connection_usable(self):
        """A rejected name must roll back so the aborted transaction does
        not poison subsequent queries (same convention as execute_query)."""
        db = _make_db()
        cursor = db.conn.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = psycopg2.errors.UndefinedTable(
            'relation "x" does not exist'
        )

        result = db.get_collection_info("no_such_table")

        assert result["success"] is False
        db.conn.rollback.assert_called_once()


@pytest.mark.live_postgres
class TestGetCollectionInfoLive:
    """Opt-in live check against a disposable PostgreSQL instance:
    EVOAGENTX_TEST_PG_DSN=postgresql://postgres:postgres@localhost:5434/postgres \
        pytest tests/src/tools/test_database_postgresql.py -m live_postgres
    """

    @pytest.mark.skipif(
        not os.environ.get("EVOAGENTX_TEST_PG_DSN"),
        reason="EVOAGENTX_TEST_PG_DSN not set",
    )
    def test_injected_payload_returns_error_and_legit_query_works(self):
        dsn = os.environ["EVOAGENTX_TEST_PG_DSN"]
        db = PostgreSQLDatabase(connection_string=dsn)
        assert not db.file_based_mode, "live test requires a real connection"

        with db.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS users")
            cur.execute("CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT)")
            cur.execute("INSERT INTO users (name) VALUES ('alice'), ('bob')")
        db.conn.commit()

        injected = db.get_collection_info(PAYLOAD)
        assert injected["success"] is False

        legit = db.get_collection_info("users")
        assert legit["success"] is True
        assert legit["data"]["row_count"] == 2
