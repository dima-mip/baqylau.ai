"""Startup login for Baqylau Student (Bento look, same palette).

Server mode is default: server URL + student login/password -> validated
against Baqylau Web before the main window opens. "Local mode" skips the
server (dev/offline, camera works immediately, no gating).
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget, QFrame)

logger = logging.getLogger(__name__)

ORG, APP = "Baqylau", "Student"

STYLESHEET = """
QDialog#LoginDialog { background: #F4F6F0; }
QLabel#LoginTitle { font-size: 26px; font-weight: 900; color: #1A1A1A; }
QLabel#LoginSub { font-size: 13px; color: #6B6B6B; }
QFrame#LoginHero { background: #1E1E1E; border-radius: 18px; }
QLabel#HeroEmoji { font-size: 34px; }
QLabel#HeroTitle { color: #FFFFFF; font-size: 19px; font-weight: 900; }
QLabel#HeroSub { color: #BBBBBB; font-size: 12px; }
QLabel#FieldLabel { font-size: 12px; font-weight: 700; color: #1A1A1A; }
QLineEdit { background: #FFFFFF; border: 2px solid #E4E4E4; border-radius: 14px;
    padding: 11px 14px; font-size: 14px; color: #1A1A1A; selection-background-color: #D8D4FF; }
QLineEdit:focus { border: 2px solid #1A1A1A; }
QLabel#LoginError { color: #C62828; font-weight: 700; font-size: 12px; }
QPushButton#LoginGo { background: #1A1A1A; color: #FFFFFF; border: none;
    border-radius: 16px; padding: 13px; font-size: 15px; font-weight: 800; }
QPushButton#LoginGo:hover { background: #333333; }
QPushButton#LoginGo:pressed { background: #000000; }
QPushButton#LoginLocal { background: transparent; color: #6B6B6B; border: none;
    padding: 8px; font-size: 13px; font-weight: 700; text-decoration: underline; }
QPushButton#LoginLocal:hover { color: #1A1A1A; }
QLabel#Chip { background: #D8D4FF; border-radius: 11px; padding: 5px 12px;
    font-size: 12px; font-weight: 700; color: #1A1A1A; }
QLabel#Chip:nth-child(2) { background: #D4F5A6; }
"""


def try_login(server: str, username: str, password: str, timeout: float = 6.0):
    """Validate credentials, return BaqylauClient or None. Never raises."""
    try:
        from core.net.client import BaqylauClient

        c = BaqylauClient(server, timeout=timeout)
        return c if c.login(username, password) else None
    except Exception as exc:
        logger.debug("login failed: %s", exc)
        return None


class LoginDialog(QDialog):
    """Student auth page shown before the main window."""

    def __init__(self, parent: QWidget | None = None, server: str = "",
                 username: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("LoginDialog")
        self.setWindowTitle("Baqylau — вход ученика")
        self.setMinimumWidth(400)
        self.setStyleSheet(STYLESHEET)
        _apply_brand_icon(self)
        self.client = None
        self.local_mode = False
        last = QSettings(ORG, APP).value("server", "http://127.0.0.1:5050")

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(26, 24, 26, 24)

        hero = QFrame()
        hero.setObjectName("LoginHero")
        hl = QHBoxLayout(hero)
        hl.setContentsMargins(18, 16, 18, 16)
        logo = QLabel()
        logo.setObjectName("HeroEmoji")
        try:
            from .branding import logo_pixmap

            pm = logo_pixmap(96)
            if pm is not None and not pm.isNull():
                logo.setPixmap(pm.scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo.setFixedSize(44, 44)
            else:
                logo.setText("🍈")
        except Exception:
            logo.setText("🍈")
        hl.addWidget(logo)
        hbox = QVBoxLayout()
        ht = QLabel("Baqylau")
        ht.setObjectName("HeroTitle")
        hs = QLabel("Локальный AI-прокторинг  •  Qostanai")
        hs.setObjectName("HeroSub")
        hbox.addWidget(ht)
        hbox.addWidget(hs)
        hl.addLayout(hbox, 1)
        lay.addWidget(hero)

        title = QLabel("Вход ученика")
        title.setObjectName("LoginTitle")
        lay.addWidget(title)
        sub = QLabel("Камера включится только после старта экзамена преподавателем.")
        sub.setObjectName("LoginSub")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        chips = QHBoxLayout()
        chips.setSpacing(8)
        for text in ("📷 Взгляд", "📱 Смартфон", "🔒 Защита"):
            ch = QLabel(text)
            ch.setObjectName("Chip")
            chips.addWidget(ch)
        chips.addStretch(1)
        lay.addLayout(chips)

        lay.addWidget(self._flabel("Сервер панели:"))
        self.ed_server = QLineEdit(server or str(last))
        self.ed_server.setPlaceholderText("http://192.168.1.10:5050")
        lay.addWidget(self.ed_server)
        lay.addWidget(self._flabel("Логин ученика:"))
        self.ed_login = QLineEdit(username)
        self.ed_login.setPlaceholderText("напр. ivan_petrov")
        lay.addWidget(self.ed_login)
        lay.addWidget(self._flabel("Пароль:"))
        self.ed_pass = QLineEdit()
        self.ed_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_pass.setPlaceholderText("••••••••")
        self.ed_pass.returnPressed.connect(self._on_login)
        lay.addWidget(self.ed_pass)

        self.error = QLabel("")
        self.error.setObjectName("LoginError")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)

        self.btn_login = QPushButton("Войти  →")
        self.btn_login.setObjectName("LoginGo")
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.clicked.connect(self._on_login)
        lay.addWidget(self.btn_login)

        self.btn_local = QPushButton("продолжить в локальном режиме (без сервера)")
        self.btn_local.setObjectName("LoginLocal")
        self.btn_local.setCursor(Qt.PointingHandCursor)
        self.btn_local.clicked.connect(self._on_local)
        lay.addWidget(self.btn_local, alignment=Qt.AlignCenter)

    @staticmethod
    def _flabel(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def _on_login(self) -> None:
        server = self.ed_server.text().strip()
        username = self.ed_login.text().strip()
        password = self.ed_pass.text()
        if not server or not username:
            self.error.setText("Укажите сервер и логин")
            return
        self.error.setText("Подключение…")
        self.error.repaint()
        c = try_login(server, username, password)
        if c is None:
            self.error.setText("Не удалось войти: проверьте сервер, логин и пароль")
            return
        QSettings(ORG, APP).setValue("server", server)
        self.client = c
        self.accept()

    def _on_local(self) -> None:
        self.local_mode = True
        self.reject()


def _apply_brand_icon(widget) -> None:
    """Set Baqylau mark as window icon. Never raises."""
    try:
        from .branding import app_icon

        icon = app_icon()
        if icon is not None and not icon.isNull():
            widget.setWindowIcon(icon)
    except Exception:
        pass
