"""Animated native desktop window bound to ReminiscenceEngine."""
from __future__ import annotations
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal, QTimer
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSizePolicy, QStackedWidget, QStatusBar, QVBoxLayout, QWidget)
from .hardware import detect_hardware
from .theme import APP_QSS

class FadeIn(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.0)
        self.setGraphicsEffect(effect)
        self._fade = QPropertyAnimation(effect, b"opacity", self)
        self._fade.setDuration(220)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
    def animate(self):
        self._fade.stop()
        self._fade.start()

class SearchBar(QLineEdit):
    submitted = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Search memories, people, places, dates...")
        self.returnPressed.connect(lambda: self.submitted.emit(self.text().strip()))

class MemoryCard(QFrame):
    def __init__(self, event, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        box = QVBoxLayout(self)
        modality = getattr(event.modality, "value", str(event.modality)).upper()
        title = QLabel(f"{modality}  •  {event.effective_event_time() or 'Unknown time'}")
        title.setObjectName("section")
        text = QLabel((event.content or "No extracted text").strip()[:420])
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(title)
        box.addWidget(text)

class ReminiscenceWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.hardware = detect_hardware(engine)
        self.setWindowTitle("Reminiscence — Private AI Memory")
        self.setMinimumSize(1050, 700)
        self.resize(1280, 800)
        self.setStyleSheet(APP_QSS)
        self._build()
        QTimer.singleShot(80, self._refresh)

    def _build(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(root)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(230)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(22, 26, 18, 20)
        brand = QLabel("REMINISCENCE")
        brand.setObjectName("brand")
        sl.addWidget(brand)
        sub = QLabel("Private multimodal memory")
        sub.setObjectName("muted")
        sl.addWidget(sub)
        sl.addSpacing(30)
        for label in ("Memories", "Timeline", "Sources", "Graph", "Settings"):
            button = QPushButton(label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, name=label: self._navigate(name))
            sl.addWidget(button)
        sl.addStretch()
        hw = QLabel(f"{self.hardware.headline}\n{self.hardware.machine} · {self.hardware.provider}")
        hw.setObjectName("muted")
        hw.setWordWrap(True)
        sl.addWidget(hw)
        layout.addWidget(side)

        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(34, 30, 34, 26)
        pl.setSpacing(18)
        title = QLabel("Your memories, connected.")
        title.setObjectName("title")
        pl.addWidget(title)
        self.search = SearchBar()
        self.search.submitted.connect(self._search)
        pl.addWidget(self.search)

        self.stack = QStackedWidget()
        self.home = self._home_page()
        self.results = self._results_page()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.results)
        pl.addWidget(self.stack, 1)
        layout.addWidget(page, 1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Local-first · no cloud dependency")

    def _home_page(self):
        page = FadeIn()
        box = QVBoxLayout(page)
        stats = QHBoxLayout()
        self.event_metric = self._metric("Memories", "—")
        self.source_metric = self._metric("Sources", "—")
        self.provider_metric = self._metric("AI provider", self.hardware.provider)
        for widget in (self.event_metric, self.source_metric, self.provider_metric):
            stats.addWidget(widget)
        box.addLayout(stats)
        heading = QLabel("Recent memories")
        heading.setObjectName("section")
        box.addWidget(heading)
        self.recent = QListWidget()
        self.recent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        box.addWidget(self.recent)
        page.animate()
        return page

    def _results_page(self):
        page = FadeIn()
        box = QVBoxLayout(page)
        self.result_label = QLabel("Search results")
        self.result_label.setObjectName("section")
        box.addWidget(self.result_label)
        self.result_list = QListWidget()
        box.addWidget(self.result_list)
        return page

    def _metric(self, label, value):
        card = QFrame()
        card.setObjectName("card")
        box = QVBoxLayout(card)
        name = QLabel(label)
        name.setObjectName("muted")
        metric = QLabel(str(value))
        metric.setObjectName("metric")
        box.addWidget(name)
        box.addWidget(metric)
        card.metric_value = metric
        return card

    def _refresh(self):
        events = self.engine.recent_memories(12)
        self.event_metric.metric_value.setText(str(len(self.engine.db.recent_events(100000))))
        self.source_metric.metric_value.setText(str(len(self.engine.sources())))
        self.recent.clear()
        for event in events:
            item = QListWidgetItem()
            card = MemoryCard(event)
            item.setSizeHint(card.sizeHint())
            self.recent.addItem(item)
            self.recent.setItemWidget(item, card)

    def _search(self, query):
        if not query:
            return
        self.result_list.clear()
        results = self.engine.search(query, top_k=12)
        self.result_label.setText(f"{len(results)} memories matching “{query}”")
        for candidate in results:
            item = QListWidgetItem()
            card = MemoryCard(candidate.event)
            item.setSizeHint(card.sizeHint())
            self.result_list.addItem(item)
            self.result_list.setItemWidget(item, card)
        self.stack.setCurrentWidget(self.results)
        self.results.animate()
        self.statusBar().showMessage(f"Local search complete · {len(results)} results")

    def _navigate(self, name):
        if name == "Memories":
            self.stack.setCurrentWidget(self.home)
            self.home.animate()
            self._refresh()
        else:
            self.statusBar().showMessage(f"{name} view is connected to the same local engine")

    def closeEvent(self, event):
        self.engine.close()
        event.accept()

def launch(engine):
    app = QApplication.instance() or QApplication([])
    window = ReminiscenceWindow(engine)
    window.show()
    return app.exec()
