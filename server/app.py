"""Baqylau Web — local teacher/admin panel + student API (Flask, LAN).

Run (teacher PC):  cd baqylau && python -m server.app  -> http://<lan-ip>:5050
First run seeds admin/admin123 (change it!). Students use the desktop app
with --server http://<lan-ip>:5050 --login ... (camera stays OFF until the
teacher starts an exam session).
"""
from __future__ import annotations

import functools
import json
import logging
import os
import time

from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

from . import db as d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baqylau")

DB = os.environ.get("BAQYLAU_DB", "data/baqylau.db")
EVIDENCE_DIR = os.environ.get("BAQYLAU_EVIDENCE", "data/evidence")
UPLOAD_DIR = os.environ.get("BAQYLAU_UPLOADS", "data/uploads")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMG_MAX_BYTES = 3 * 1024 * 1024

MAIN_VIOLATIONS = ("face_missing", "phone", "phone_raised", "reading_device",
                   "head_sustained", "gaze_sustained", "focus_lost", "shortcut")


def create_app(db_path: str = DB) -> Flask:
    d.init_db(db_path)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app = Flask(__name__, template_folder="templates")
    app.secret_key = os.environ.get("BAQYLAU_SECRET", "baqylau-local")
    app.config["DB"] = db_path

    def con():
        return d.connect(app.config["DB"])

    def me():
        u = session.get("user")
        return u

    def login_required(*roles):
        def deco(fn):
            @functools.wraps(fn)
            def wrap(*a, **k):
                u = me()
                if not u:
                    return redirect(url_for("login"))
                if roles and u["role"] not in roles:
                    return "Forbidden", 403
                return fn(*a, **k)
            return wrap
        return deco

    def api_user():
        tok = request.headers.get("X-Token", "")
        if not tok:
            return None
        c = con()
        try:
            return d.user_by_token(c, tok)
        finally:
            c.close()

    # ---------- web: auth ----------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            c = con()
            try:
                u = d.verify_user(c, request.form.get("username", ""),
                                  request.form.get("password", ""))
            finally:
                c.close()
            if u and u["role"] in ("teacher", "admin"):
                session["user"] = {"id": u["id"], "username": u["username"],
                                   "role": u["role"], "name": u["full_name"]}
                return redirect(url_for("dashboard"))
            return render_template("login.html", error="Неверный логин/пароль")
        return render_template("login.html", error="")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    def index():
        return redirect(url_for("dashboard"))

    # ---------- web: dashboard ----------
    @app.route("/dashboard")
    @login_required("teacher", "admin")
    def dashboard():
        c = con()
        try:
            exams = c.execute("SELECT * FROM exams ORDER BY id DESC").fetchall()
            groups = c.execute("SELECT * FROM groups ORDER BY name").fetchall()
            sess = c.execute(
                "SELECT s.*, e.title AS exam_title, g.name AS group_name FROM sessions s "
                "LEFT JOIN exams e ON e.id=s.exam_id LEFT JOIN groups g ON g.id=s.group_id "
                "ORDER BY s.id DESC LIMIT 5").fetchall()
            unseen = c.execute("SELECT COUNT(*) n FROM notifications WHERE seen=0").fetchone()["n"]
        finally:
            c.close()
        return render_template("dashboard.html", user=me(), exams=exams, groups=groups,
                               sessions=sess, unseen=unseen)

    @app.route("/api/teacher/state")
    def teacher_state():
        u = me()
        if not u or u["role"] not in ("teacher", "admin"):
            return jsonify({"error": "auth"}), 401
        c = con()
        try:
            studs = c.execute(
                "SELECT u.id,u.full_name,u.username,g.name AS grp FROM users u "
                "LEFT JOIN groups g ON g.id=u.group_id WHERE u.role='student' "
                "ORDER BY g.name,u.full_name").fetchall()
            hb = {r["student_id"]: dict(r) for r in
                  c.execute("SELECT * FROM heartbeats").fetchall()}
            sess = c.execute(
                "SELECT * FROM sessions WHERE status='active' ORDER BY id DESC").fetchall()
            active_groups = {s["group_id"]: s for s in sess}
            cards = []
            for s in studs:
                h = hb.get(s["id"], {})
                viol = d.jload(h.get("violations_json", "[]"), [])
                main = [v for v in viol if v.get("kind") in MAIN_VIOLATIONS][:4]
                ses = active_groups.get(_group_id(c, s["id"]))
                cards.append({
                    "id": s["id"],
                    "name": s["full_name"] or s["username"], "group": s["grp"] or "—",
                    "risk": h.get("risk", 0), "level": h.get("level", "—"),
                    "violations": main, "thumb": h.get("thumb_b64", ""),
                    "ago": _ago(h.get("updated_at", 0)),
                    "exam": _exam_title(c, ses["exam_id"]) if ses else "—",
                    "session": "active" if ses else "idle",
                })
            unseen = c.execute("SELECT COUNT(*) n FROM notifications WHERE seen=0").fetchone()["n"]
            s0 = sess[0] if sess else None
            recent = c.execute(
                "SELECT ts,student AS who,detail AS text,kind FROM notifications "
                "ORDER BY id DESC LIMIT 8").fetchall()
            return jsonify({
                "cards": cards, "unseen": unseen, "active": bool(sess),
                "session": ({"active": True, "title": _exam_title(c, s0["exam_id"]),
                             "started_at": s0["started_at"]} if s0
                            else {"active": False}),
                "recent": [dict(r) for r in recent],
            })
        finally:
            c.close()

    def _group_id(c, student_id):
        r = c.execute("SELECT group_id FROM users WHERE id=?", (student_id,)).fetchone()
        return r["group_id"] if r else None

    def _exam_title(c, exam_id):
        r = c.execute("SELECT title FROM exams WHERE id=?", (exam_id,)).fetchone()
        return r["title"] if r else "—"

    def _ago(ts):
        if not ts:
            return "—"
        s = int(time.time() - ts)
        return f"{s}с" if s < 60 else f"{s // 60}м"

    # ---------- web: groups/students ----------
    @app.route("/groups", methods=["GET", "POST"])
    @login_required("teacher", "admin")
    def groups():
        c = con()
        try:
            if request.method == "POST":
                if request.form.get("action") == "group":
                    c.execute("INSERT OR IGNORE INTO groups(name,description) VALUES(?,?)",
                              (request.form.get("name", "").strip(),
                               request.form.get("description", "")))
                    c.commit()
                elif request.form.get("action") == "student":
                    import secrets as _s

                    salt = _s.token_hex(8)
                    c.execute(
                        "INSERT INTO users(username,pass_hash,salt,role,full_name,group_id)"
                        " VALUES(?,?,?,?,?,?)",
                        (request.form.get("username", "").strip(),
                         d.hash_password(request.form.get("password", "1234"), salt),
                         salt, "student", request.form.get("full_name", ""),
                         int(request.form.get("group_id") or 0) or None))
                    c.commit()
                elif request.form.get("action") == "del_student":
                    c.execute("DELETE FROM users WHERE id=? AND role='student'",
                              (int(request.form.get("id", 0)),))
                    c.commit()
                return redirect(url_for("groups"))
            groups = c.execute("SELECT * FROM groups ORDER BY name").fetchall()
            students = c.execute(
                "SELECT u.*,g.name AS grp FROM users u LEFT JOIN groups g ON g.id=u.group_id "
                "WHERE u.role='student' ORDER BY g.name,u.full_name").fetchall()
            return render_template("groups.html", user=me(), groups=groups, students=students)
        finally:
            c.close()

    # ---------- web: exams ----------
    @app.route("/exams")
    @login_required("teacher", "admin")
    def exams():
        c = con()
        try:
            exams = c.execute("SELECT * FROM exams ORDER BY id DESC").fetchall()
            out = []
            for e in exams:
                q = d.jload(e["questions_json"], [])
                out.append({"row": e, "n": len(q)})
            return render_template("exams.html", user=me(), exams=out)
        finally:
            c.close()

    @app.route("/exams/new", methods=["GET", "POST"])
    @login_required("teacher", "admin")
    def exam_new():
        if request.method == "POST":
            try:
                qs = json.loads(request.form.get("questions_json", "[]"))
                assert isinstance(qs, list) and all(
                    q.get("q", "").strip() and
                    len([o for o in q.get("options", []) if str(o).strip()]) >= 2
                    for q in qs)
            except Exception:
                return render_template("exam_form.html", user=me(),
                                       error="У каждого вопроса нужен текст и минимум 2 варианта")
            # question images: qimg_<idx> files
            import secrets as _s

            for i, q in enumerate(qs):
                f = request.files.get(f"qimg_{i}")
                if f and f.filename:
                    ext = os.path.splitext(f.filename)[1].lower()
                    if ext in IMG_EXTS:
                        data = f.read()
                        if data and len(data) <= IMG_MAX_BYTES:
                            name = f"q_{_s.token_hex(6)}{ext}"
                            with open(os.path.join(UPLOAD_DIR, name), "wb") as fh:
                                fh.write(data)
                            q["image"] = name
            c = con()
            c = con()
            try:
                c.execute(
                    "INSERT INTO exams(title,subject,duration_min,questions_json,created_by,created_at)"
                    " VALUES(?,?,?,?,?,?)",
                    (request.form.get("title", "Без названия"), request.form.get("subject", ""),
                     int(request.form.get("duration_min", 30) or 30),
                     json.dumps(qs, ensure_ascii=False), me()["id"], time.time()))
                c.commit()
                d.audit(c, me()["username"], "exam_create", request.form.get("title", ""))
            finally:
                c.close()
            return redirect(url_for("exams"))
        return render_template("exam_form.html", user=me(), error="")

    @app.route("/exams/<int:eid>/delete", methods=["POST"])
    @login_required("teacher", "admin")
    def exam_del(eid):
        c = con()
        try:
            c.execute("DELETE FROM exams WHERE id=?", (eid,))
            c.commit()
        finally:
            c.close()
        return redirect(url_for("exams"))

    # ---------- web: sessions ----------
    @app.route("/session/start", methods=["POST"])
    @login_required("teacher", "admin")
    def session_start():
        c = con()
        try:
            c.execute("UPDATE sessions SET status='finished',finished_at=? WHERE status='active'",
                      (time.time(),))
            c.execute(
                "INSERT INTO sessions(exam_id,group_id,status,started_at,started_by)"
                " VALUES(?,?,'active',?,?)",
                (int(request.form.get("exam_id")), int(request.form.get("group_id")),
                 time.time(), me()["id"]))
            c.commit()
            d.audit(c, me()["username"], "session_start",
                    f"exam={request.form.get('exam_id')} group={request.form.get('group_id')}")
        finally:
            c.close()
        return redirect(url_for("dashboard"))

    @app.route("/session/stop", methods=["POST"])
    @login_required("teacher", "admin")
    def session_stop():
        c = con()
        try:
            c.execute("UPDATE sessions SET status='finished',finished_at=? WHERE status='active'",
                      (time.time(),))
            c.commit()
            d.audit(c, me()["username"], "session_stop", "")
        finally:
            c.close()
        return redirect(url_for("dashboard"))

    # ---------- web: audit log (admin, full info) ----------
    @app.route("/audit")
    @login_required("admin")
    def audit_log():
        c = con()
        try:
            rows = c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 300").fetchall()
            evs = c.execute(
                "SELECT e.*, COALESCE(u.full_name, u.username, 'ученик #' || e.student_id) AS student "
                "FROM evidence e LEFT JOIN users u ON u.id=e.student_id "
                "ORDER BY e.id DESC LIMIT 200"
            ).fetchall()
            subs = c.execute(
                "SELECT s.*, COALESCE(u.full_name, u.username, 'ученик #' || s.student_id) AS student, "
                "COALESCE(ex.title, '—') AS exam FROM submissions s "
                "LEFT JOIN users u ON u.id=s.student_id "
                "LEFT JOIN sessions se ON se.id=s.session_id "
                "LEFT JOIN exams ex ON ex.id=se.exam_id "
                "ORDER BY s.id DESC LIMIT 200").fetchall()
            return render_template("audit.html", user=me(), rows=rows, evs=evs, subs=subs)
        finally:
            c.close()

    # ---------- web: users (admin) ----------
    @app.route("/users", methods=["GET", "POST"])
    @login_required("admin")
    def users():
        c = con()
        try:
            if request.method == "POST":
                import secrets as _s

                salt = _s.token_hex(8)
                c.execute(
                    "INSERT INTO users(username,pass_hash,salt,role,full_name) VALUES(?,?,?,?,?)",
                    (request.form.get("username", "").strip(),
                     d.hash_password(request.form.get("password", "1234"), salt),
                     salt, request.form.get("role", "teacher"),
                     request.form.get("full_name", "")))
                c.commit()
                return redirect(url_for("users"))
            users = c.execute(
                "SELECT id,username,role,full_name FROM users ORDER BY role,username").fetchall()
            return render_template("users.html", user=me(), users=users)
        finally:
            c.close()

    @app.route("/users/<int:uid>/delete", methods=["POST"])
    @login_required("admin")
    def user_del(uid):
        c = con()
        try:
            me_id = me()["id"]
            if uid == me_id:
                return "Нельзя удалить себя", 400
            target = c.execute("SELECT role FROM users WHERE id=?", (uid,)).fetchone()
            if target is None:
                return redirect(url_for("users"))
            if target["role"] == "admin":
                n = c.execute("SELECT COUNT(*) n FROM users WHERE role='admin'").fetchone()["n"]
                if n <= 1:
                    return "Нельзя удалить последнего админа", 400
            c.execute("DELETE FROM users WHERE id=?", (uid,))
            c.commit()
            d.audit(c, me()["username"], "user_delete", str(uid))
        finally:
            c.close()
        return redirect(url_for("users"))

    # ---------- web: notifications + evidence ----------
    @app.route("/notifications", methods=["GET", "POST"])
    @login_required("teacher", "admin")
    def notifications():
        c = con()
        try:
            if request.method == "POST":
                c.execute("UPDATE notifications SET seen=1")
                c.commit()
                return redirect(url_for("notifications"))
            notes = c.execute(
                "SELECT n.*, e.path AS epath FROM notifications n "
                "LEFT JOIN evidence e ON e.id=n.evidence_id "
                "ORDER BY n.id DESC LIMIT 200").fetchall()
            return render_template("notifications.html", user=me(), notes=notes)
        finally:
            c.close()

    @app.route("/notifications/<int:nid>/delete", methods=["POST"])
    @login_required("teacher", "admin")
    def notification_del(nid):
        c = con()
        try:
            c.execute("DELETE FROM notifications WHERE id=?", (nid,))
            c.commit()
            d.audit(c, me()["username"], "notification_delete", str(nid))
            if request.is_json or request.headers.get("X-Requested-With") == "fetch":
                return jsonify({"ok": True})
            return redirect(url_for("notifications"))
        finally:
            c.close()

    @app.route("/evidence/<path:name>")
    @login_required("teacher", "admin")
    def evidence_file(name):
        return send_from_directory(os.path.abspath(EVIDENCE_DIR), name)

    @app.route("/uploads/<path:name>")
    def upload_file(name):
        """Question images: web session or student token required."""
        u = me()
        if not u:
            tok = request.headers.get("X-Token", "")
            c0 = con()
            try:
                u = d.user_by_token(c0, tok)
            finally:
                c0.close()
        if not u:
            return "Forbidden", 403
        return send_from_directory(os.path.abspath(UPLOAD_DIR), name)

    @app.template_filter("dt")
    def _dt(ts):
        try:
            return time.strftime("%d.%m %H:%M:%S", time.localtime(float(ts)))
        except Exception:
            return "—"

    @app.route("/api/warn", methods=["POST"])
    def api_warn():
        u = me()
        if not u or u["role"] not in ("teacher", "admin"):
            return jsonify({"error": "auth"}), 401
        data = request.get_json(force=True, silent=True) or {}
        try:
            sid = int(data.get("student_id", 0))
        except Exception:
            return jsonify({"error": "bad id"}), 400
        c = con()
        try:
            st = c.execute("SELECT full_name,username FROM users WHERE id=? AND role='student'",
                           (sid,)).fetchone()
            if not st:
                return jsonify({"error": "no student"}), 404
            who = st["full_name"] or st["username"]
            text = f"Предупреждение от преподавателя ({u['username']})"
            c.execute("INSERT INTO warnings(student_id,text,ts) VALUES(?,?,?)",
                      (sid, text, time.time()))
            c.execute("INSERT INTO notifications(ts,student,kind,detail) VALUES(?,?,?,?)",
                      (time.time(), who, "warn", f"{who}: {text.lower()}"))
            c.commit()
            d.audit(c, u["username"], "warn", who)
            return jsonify({"ok": True})
        finally:
            c.close()

    # ---------- student API ----------
    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(force=True, silent=True) or {}
        c = con()
        try:
            u = d.verify_user(c, data.get("username", ""), data.get("password", ""))
            if not u or u["role"] != "student":
                return jsonify({"error": "auth"}), 401
            tok = d.make_token(c, u["id"])
            return jsonify({"token": tok, "name": u["full_name"] or u["username"],
                            "group_id": u["group_id"]})
        finally:
            c.close()

    def _student_session(c, student_id):
        g = c.execute("SELECT group_id FROM users WHERE id=?", (student_id,)).fetchone()
        if not g or not g["group_id"]:
            return None
        return c.execute(
            "SELECT s.*,e.title,e.subject,e.duration_min FROM sessions s "
            "JOIN exams e ON e.id=s.exam_id WHERE s.group_id=? "
            "ORDER BY s.id DESC LIMIT 1", (g["group_id"],)).fetchone()

    @app.route("/api/session")
    def api_session():
        u = api_user()
        if not u or u["role"] != "student":
            return jsonify({"error": "auth"}), 401
        c = con()
        try:
            s = _student_session(c, u["id"])
            if not s:
                return jsonify({"status": "idle"})
            warns = c.execute(
                "SELECT id,text FROM warnings WHERE student_id=? AND delivered=0 "
                "ORDER BY id", (u["id"],)).fetchall()
            if warns:
                c.execute("UPDATE warnings SET delivered=1 WHERE student_id=?", (u["id"],))
                c.commit()
            return jsonify({"status": s["status"], "session_id": s["id"],
                            "exam_id": s["exam_id"], "title": s["title"],
                            "subject": s["subject"], "duration_min": s["duration_min"],
                            "warnings": [w["text"] for w in warns]})
        finally:
            c.close()

    @app.route("/api/exam")
    def api_exam():
        u = api_user()
        if not u or u["role"] != "student":
            return jsonify({"error": "auth"}), 401
        c = con()
        try:
            s = _student_session(c, u["id"])
            if not s or s["status"] != "active":
                return jsonify({"error": "no active exam"}), 404
            e = c.execute("SELECT * FROM exams WHERE id=?", (s["exam_id"],)).fetchone()
            qs = d.jload(e["questions_json"], [])
            public = [{"q": q.get("q", ""), "options": q.get("options", []),
                       "image": ("/uploads/" + q["image"]) if q.get("image") else None}
                      for q in qs]
            return jsonify({"exam_id": e["id"], "title": e["title"],
                            "duration_min": e["duration_min"], "questions": public})
        finally:
            c.close()

    @app.route("/api/submit", methods=["POST"])
    def api_submit():
        u = api_user()
        if not u or u["role"] != "student":
            return jsonify({"error": "auth"}), 401
        data = request.get_json(force=True, silent=True) or {}
        c = con()
        try:
            s = _student_session(c, u["id"])
            if not s or s["status"] != "active":
                return jsonify({"error": "no active exam"}), 404
            e = c.execute("SELECT * FROM exams WHERE id=?", (s["exam_id"],)).fetchone()
            qs = d.jload(e["questions_json"], [])
            ans = data.get("answers", [])
            score = sum(1 for i, q in enumerate(qs)
                        if i < len(ans) and ans[i] == q.get("correct", -1))
            c.execute(
                "INSERT OR REPLACE INTO submissions(session_id,student_id,score,total,answers_json,submitted_at)"
                " VALUES(?,?,?,?,?,?)",
                (s["id"], u["id"], score, len(qs), json.dumps(ans), time.time()))
            c.commit()
            return jsonify({"score": score, "total": len(qs)})
        finally:
            c.close()

    @app.route("/api/heartbeat", methods=["POST"])
    def api_heartbeat():
        u = api_user()
        if not u or u["role"] != "student":
            return jsonify({"error": "auth"}), 401
        data = request.get_json(force=True, silent=True) or {}
        thumb = str(data.get("thumb", ""))[:120_000]  # cap: protect teacher PC
        viol = data.get("violations", [])[:8]
        c = con()
        try:
            c.execute(
                "INSERT OR REPLACE INTO heartbeats(student_id,risk,level,violations_json,thumb_b64,updated_at)"
                " VALUES(?,?,?,?,?,?)",
                (u["id"], float(data.get("risk", 0)), str(data.get("level", "OK"))[:12],
                 json.dumps(viol, ensure_ascii=False)[:4000], thumb, time.time()))
            c.commit()
            return jsonify({"ok": True})
        finally:
            c.close()

    @app.route("/api/evidence", methods=["POST"])
    def api_evidence():
        u = api_user()
        if not u or u["role"] != "student":
            return jsonify({"error": "auth"}), 401
        f = request.files.get("file")
        kind = request.form.get("kind", "photo")
        risk = float(request.form.get("risk", 0))
        session_id = int(request.form.get("session_id", 0) or 0)
        if not f:
            return jsonify({"error": "no file"}), 400
        name = f"ev_{u['id']}_{int(time.time())}_{kind}{os.path.splitext(f.filename)[1][:5]}"
        path = os.path.join(EVIDENCE_DIR, name)
        f.save(path)
        try:
            from .drive import upload_if_configured

            link = upload_if_configured(path)
        except Exception:
            link = ""
        c = con()
        try:
            cur = c.execute(
                "INSERT INTO evidence(student_id,session_id,kind,path,drive_link,risk,ts)"
                " VALUES(?,?,?,?,?,?,?)",
                (u["id"], session_id, kind, name, link, risk, time.time()))
            evid = cur.lastrowid
            who = u["full_name"] or u["username"]
            c.execute(
                "INSERT INTO notifications(ts,student,kind,detail,evidence_id) VALUES(?,?,?,?,?)",
                (time.time(), who, "RISK100" if risk >= 100 else kind,
                 f"{who}: риск {risk:.0f} ({kind})" + (" [Drive]" if link else " [локально]"),
                 evid))
            c.commit()
            return jsonify({"ok": True, "id": evid, "drive": bool(link)})
        finally:
            c.close()

    return app


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Baqylau Web (teacher/admin panel)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--db", default=DB)
    a = p.parse_args()
    create_app(a.db).run(host=a.host, port=a.port, threaded=True)
