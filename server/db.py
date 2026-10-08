"""Baqylau Web: local SQLite storage (stdlib only).

Tables: users (teacher/admin/student), groups, exams, sessions,
submissions, heartbeats (latest student state), evidence, notifications,
tokens, audit.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: str) -> None:
    con = connect(db_path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL, pass_hash TEXT NOT NULL, salt TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'student', full_name TEXT DEFAULT '',
      group_id INTEGER);
    CREATE TABLE IF NOT EXISTS groups(
      id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
      description TEXT DEFAULT '');
    CREATE TABLE IF NOT EXISTS exams(
      id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
      subject TEXT DEFAULT '', duration_min INTEGER DEFAULT 30,
      questions_json TEXT NOT NULL DEFAULT '[]',
      created_by INTEGER, created_at REAL);
    CREATE TABLE IF NOT EXISTS sessions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, exam_id INTEGER NOT NULL,
      group_id INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'active',
      started_at REAL, finished_at REAL, started_by INTEGER);
    CREATE TABLE IF NOT EXISTS submissions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
      student_id INTEGER NOT NULL, score INTEGER DEFAULT 0, total INTEGER DEFAULT 0,
      answers_json TEXT DEFAULT '[]', submitted_at REAL,
      UNIQUE(session_id, student_id));
    CREATE TABLE IF NOT EXISTS heartbeats(
      student_id INTEGER PRIMARY KEY, risk REAL DEFAULT 0, level TEXT DEFAULT 'OK',
      violations_json TEXT DEFAULT '[]', thumb_b64 TEXT DEFAULT '', updated_at REAL);
    CREATE TABLE IF NOT EXISTS evidence(
      id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, session_id INTEGER,
      kind TEXT DEFAULT 'photo', path TEXT DEFAULT '', drive_link TEXT DEFAULT '',
      risk REAL DEFAULT 0, ts REAL);
    CREATE TABLE IF NOT EXISTS notifications(
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, student TEXT DEFAULT '',
      kind TEXT DEFAULT '', detail TEXT DEFAULT '', evidence_id INTEGER, seen INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS tokens(
      token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at REAL);
    CREATE TABLE IF NOT EXISTS warnings(
      id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER NOT NULL,
      text TEXT DEFAULT '', ts REAL, delivered INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS audit(
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT DEFAULT '',
      kind TEXT DEFAULT '', detail TEXT DEFAULT '');
    """)
    con.commit()
    # seed admin/admin123 on first run
    row = cur.execute("SELECT id FROM users WHERE username='admin'").fetchone()
    if row is None:
        salt = secrets.token_hex(8)
        cur.execute(
            "INSERT INTO users(username,pass_hash,salt,role,full_name) VALUES(?,?,?,?,?)",
            ("admin", hash_password("admin123", salt), salt, "admin", "Administrator"))
        con.commit()
        print("Baqylau: seeded admin / admin123 — CHANGE IT after first login!")
    con.close()


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def verify_user(con: sqlite3.Connection, username: str, password: str):
    row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row is None:
        return None
    if hash_password(password, row["salt"]) == row["pass_hash"]:
        return row
    return None


def make_token(con: sqlite3.Connection, user_id: int) -> str:
    tok = secrets.token_hex(24)
    con.execute("INSERT OR REPLACE INTO tokens(token,user_id,created_at) VALUES(?,?,?)",
                (tok, user_id, time.time()))
    con.commit()
    return tok


def user_by_token(con: sqlite3.Connection, token: str):
    row = con.execute(
        "SELECT u.* FROM users u JOIN tokens t ON t.user_id=u.id WHERE t.token=?",
        (token,)).fetchone()
    return row


def audit(con: sqlite3.Connection, actor: str, kind: str, detail: str) -> None:
    try:
        con.execute("INSERT INTO audit(ts,actor,kind,detail) VALUES(?,?,?,?)",
                    (time.time(), actor, kind, detail))
        con.commit()
    except Exception:
        pass


def jload(s: str, default):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default
