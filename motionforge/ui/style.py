"""Dark theme stylesheet for the MotionForge UI."""

ACCENT = "#4fc3f7"
ACCENT_DIM = "#2a6f8f"
GREEN = "#69f0ae"
RED = "#ff5252"
AMBER = "#ffd740"

QSS = """
QMainWindow, QDialog, QWizard { background: #14181d; }
QWidget { color: #e8eaed; font-size: 13px; font-family: 'Segoe UI'; }
QGroupBox {
    border: 1px solid #2a3138; border-radius: 8px; margin-top: 12px; padding-top: 10px;
    background: #1a2027;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9fb3c8; }
QTabWidget::pane { border: 1px solid #2a3138; border-radius: 6px; background: #1a2027; }
QTabBar::tab {
    background: #1a2027; color: #9fb3c8; padding: 7px 18px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px;
}
QTabBar::tab:selected { background: #232b34; color: #4fc3f7; }
QPushButton {
    background: #232b34; border: 1px solid #2f3942; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover { background: #2b3540; border-color: #4fc3f7; }
QPushButton:pressed { background: #1d242c; }
QPushButton:disabled { color: #5c6770; }
QPushButton#armButton {
    font-weight: 700; font-size: 14px; padding: 8px 22px; border-radius: 8px;
    background: #1d3a2a; border: 1px solid #2e7d32; color: #69f0ae;
}
QPushButton#armButton:checked { background: #402020; border-color: #b71c1c; color: #ff8a80; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #10141a; border: 1px solid #2f3942; border-radius: 5px; padding: 4px 8px;
    selection-background-color: #2a6f8f;
}
QComboBox QAbstractItemView { background: #1a2027; border: 1px solid #2f3942; }
QTableWidget {
    background: #10141a; gridline-color: #232b34; border: 1px solid #2a3138; border-radius: 6px;
}
QTableWidget::item { padding: 4px; }
QHeaderView::section {
    background: #1a2027; color: #9fb3c8; border: none; padding: 6px; font-weight: 600;
}
QListWidget { background: #10141a; border: 1px solid #2a3138; border-radius: 6px; }
QSlider::groove:horizontal { height: 5px; background: #2a3138; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: #4fc3f7;
}
QProgressBar {
    background: #10141a; border: 1px solid #2a3138; border-radius: 6px; text-align: center;
}
QProgressBar::chunk { background: #4fc3f7; border-radius: 5px; }
QStatusBar { background: #10141a; color: #9fb3c8; }
QLabel#statTitle { color: #9fb3c8; font-size: 11px; }
QLabel#statValue { color: #4fc3f7; font-size: 17px; font-weight: 700; }
QLabel#gameName { font-size: 16px; font-weight: 700; color: #e8eaed; }
QLabel#appTitle { font-size: 18px; font-weight: 800; color: #4fc3f7; letter-spacing: 1px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QToolTip { background: #232b34; color: #e8eaed; border: 1px solid #4fc3f7; }
"""
