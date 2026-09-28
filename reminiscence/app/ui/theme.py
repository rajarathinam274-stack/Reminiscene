"""Compact, GPU-friendly visual system."""
APP_QSS = """
QMainWindow,QWidget{background:#0b0f14;color:#e7edf5;font-family:'Segoe UI',sans-serif}
QFrame#sidebar{background:#10161e;border-right:1px solid #1e2a36}
QLabel#brand{font-size:24px;font-weight:700;letter-spacing:1px}
QLabel#muted{color:#8190a3} QLabel#title{font-size:28px;font-weight:700}
QLabel#section{font-size:14px;font-weight:600;color:#aab8c8}
QLineEdit{background:#111923;border:1px solid #263442;border-radius:12px;padding:12px;color:#eef5fc}
QLineEdit:focus{border:1px solid #4d9cff}
QPushButton{background:#16212c;border:1px solid #263442;border-radius:10px;padding:10px 14px;color:#dce7f2}
QPushButton:hover{background:#1d2b38}
QFrame#card{background:#111923;border:1px solid #1e2b38;border-radius:16px}
QLabel#metric{font-size:22px;font-weight:700}
QListWidget{background:transparent;border:0}
QListWidget::item{background:#111923;border:1px solid #1e2b38;border-radius:12px;padding:10px;margin:4px 0}
QStatusBar{background:#0d131a;color:#8190a3}
"""
