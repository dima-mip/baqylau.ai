"""Post-exam HTML report: timeline + snapshot gallery + verdict (§4).

Reads ``data/logs/events.jsonl`` (written by AlertManager) and snapshot
images, emits a single self-contained-viewable HTML file. Stdlib only.

Usage:
    python tools_report.py [--log-dir data/logs] [--student NAME]
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def load_events(log_dir: str) -> list[dict]:
    path = os.path.join(log_dir, "events.jsonl")
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return out


def verdict(events: list[dict]) -> tuple[str, str]:
    """Return (verdict, color)."""
    if not events:
        return "PASS — нарушений нет", "#2E7D32"
    worst = max((e.get("level", "LOW") for e in events),
                key=lambda l: LEVEL_ORDER.get(l, 0))
    if worst == "CRITICAL":
        return "FAIL — критичные нарушения, требуется проверка", "#C62828"
    if worst == "HIGH":
        return "REVIEW — есть серьёзные нарушения", "#EF6C00"
    return "PASS с замечаниями (LOW/MEDIUM)", "#2E7D32"


def snapshots(log_dir: str) -> list[str]:
    pats = [os.path.join(log_dir, "snapshots", "CRITICAL_*"),
            os.path.join(log_dir, "snapshots", "SCREEN_*")]
    files = []
    for p in pats:
        files.extend(glob.glob(p + ".jpg") + glob.glob(p + ".png"))
    return sorted(files)


def build_html(student: str, events: list[dict], shots: list[str], log_dir: str) -> str:
    ver, color = verdict(events)
    counts: dict[str, int] = {}
    for e in events:
        counts[e.get("level", "?")] = counts.get(e.get("level", "?"), 0) + 1
    max_score = max([float(e.get("score", 0)) for e in events] or [0.0])

    def esc(v) -> str:
        return html.escape(str(v))

    rows = "\n".join(
        "<tr><td>{t}</td><td><b style='color:{c}'>{l}</b></td>"
        "<td>{k}</td><td>{s}</td><td>{d}</td></tr>".format(
            t=esc(e.get("time", "")), c="#7B0000" if e.get("level") == "CRITICAL" else "#1A1A1A",
            l=esc(e.get("level", "")), k=esc(e.get("kind", "")),
            s=esc(e.get("score", "")), d=esc(e.get("detail", "")))
        for e in events
    ) or "<tr><td colspan=5>Нарушений не зафиксировано</td></tr>"
    imgs = "\n".join(
        f"<figure><img src='{esc(os.path.relpath(p, log_dir))}' loading='lazy'>"
        f"<figcaption>{esc(os.path.basename(p))}</figcaption></figure>"
        for p in shots
    ) or "<p>Снапшотов нет</p>"
    counts_html = " · ".join(f"{esc(k)}: {v}" for k, v in sorted(counts.items())) or "—"
    return f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Отчёт Baqylau — {esc(student)}</title>
<style>body{{font-family:'Segoe UI',sans-serif;background:#F4F6F0;color:#1A1A1A;max-width:1100px;margin:24px auto;padding:0 16px}}
.card{{background:#fff;border-radius:18px;padding:18px;margin:14px 0}}
.verdict{{font-size:22px;font-weight:800;color:{color}}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}
figure{{display:inline-block;margin:8px}}img{{max-width:320px;border-radius:12px;border:1px solid #ccc}}
</style></head><body>
<h1>Отчёт Baqylau — {esc(student)}</h1>
<div class="card"><div class="verdict">{esc(ver)}</div>
<p>Событий: {len(events)} ({counts_html}) · Макс. риск: {max_score:.0f} ·
Сформирован: {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}</p></div>
<div class="card"><h2>Таймлайн нарушений</h2>
<table><tr><th>Время</th><th>Уровень</th><th>Тип</th><th>RISK</th><th>Детали</th></tr>{rows}</table></div>
<div class="card"><h2>Кадры нарушений ({len(shots)})</h2>{imgs}</div>
</body></html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Post-exam proctoring report")
    p.add_argument("--log-dir", default="data/logs")
    p.add_argument("--student", default="Student")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    events = load_events(a.log_dir)
    shots = snapshots(a.log_dir)
    out = a.out or os.path.join(
        a.log_dir, f"report_{time.strftime('%Y%m%d_%H%M%S')}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(a.student, events, shots, a.log_dir))
    print(f"Report: {out} ({len(events)} events, {len(shots)} snapshots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
