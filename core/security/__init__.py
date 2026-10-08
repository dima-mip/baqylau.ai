"""Workstation protection subpackage (hackathon case §2.3).

HotkeyGuard (``keyboard``) suppresses exam-banned combos, FocusGuard
(``pygetwindow``) flags leaving the exam window. Both degrade gracefully
to disabled state when optional deps are missing (non-Windows, CI).
"""
