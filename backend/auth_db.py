from pathlib import Path
import sqlite3
from contextlib import contextmanager

from werkzeug.security import generate_password_hash, check_password_hash


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "metadata" / "users.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


@contextmanager
def db_connection():
    connection = get_connection()

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_auth_db():
    """Create authentication tables if they do not already exist."""

    with db_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('admin', 'user')),
                active_dataset_id TEXT NOT NULL DEFAULT 'online_retail',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
        """)

        connection.execute("""
            INSERT OR IGNORE INTO app_settings (
                setting_key, setting_value
            )
            VALUES ('registration_open', '0')
        """)


def create_user(name, email, password, role="user"):
    """Create a user with a securely hashed password."""

    if role not in ("admin", "user"):
        raise ValueError("Invalid user role.")

    password_hash = generate_password_hash(password)

    with db_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO users (name, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """, (
            name.strip(),
            email.strip().lower(),
            password_hash,
            role
        ))

        return cursor.lastrowid


def get_user_by_email(email):
    with db_connection() as connection:
        row = connection.execute("""
            SELECT * FROM users
            WHERE email = ?
        """, (email.strip().lower(),)).fetchone()

        return dict(row) if row else None


def get_user_by_id(user_id):
    with db_connection() as connection:
        row = connection.execute("""
            SELECT * FROM users
            WHERE id = ?
        """, (user_id,)).fetchone()

        return dict(row) if row else None


def verify_password(user, password):
    if not user:
        return False

    return check_password_hash(
        user["password_hash"],
        password
    )


def count_admins():
    with db_connection() as connection:
        row = connection.execute("""
            SELECT COUNT(*) AS total
            FROM users
            WHERE role = 'admin'
        """).fetchone()

        return row["total"]


def get_active_dataset(user_id):
    with db_connection() as connection:
        row = connection.execute("""
            SELECT active_dataset_id
            FROM users
            WHERE id = ?
        """, (user_id,)).fetchone()

        if row is None:
            raise ValueError("User not found.")

        return row["active_dataset_id"]


def set_active_dataset(user_id, dataset_id):
    with db_connection() as connection:
        cursor = connection.execute("""
            UPDATE users
            SET active_dataset_id = ?
            WHERE id = ?
        """, (dataset_id, user_id))

        if cursor.rowcount == 0:
            raise ValueError("User not found.")


def get_setting(key, default=None):
    with db_connection() as connection:
        row = connection.execute("""
            SELECT setting_value
            FROM app_settings
            WHERE setting_key = ?
        """, (key,)).fetchone()

        return row["setting_value"] if row else default


def set_setting(key, value):
    with db_connection() as connection:
        connection.execute("""
            INSERT INTO app_settings (setting_key, setting_value)
            VALUES (?, ?)
            ON CONFLICT(setting_key)
            DO UPDATE SET setting_value = excluded.setting_value
        """, (key, str(value)))
