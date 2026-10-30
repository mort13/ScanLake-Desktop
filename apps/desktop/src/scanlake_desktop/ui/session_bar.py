"""Top bar: ship, location, session start/close and the rock cluster."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from scanlake_core.domain.tables import tables


class SessionBar(QWidget):
    ship_changed = Signal(str)
    location_changed = Signal(str, str, str)  # system, gravity well, region
    start_clicked = Signal()
    close_clicked = Signal()
    cluster_step = Signal(int)
    new_cluster_clicked = Signal()
    account_clicked = Signal()

    def __init__(self, ships: list[tuple[str, str]]) -> None:
        super().__init__()
        self.ship = QComboBox()
        for key, label in ships:
            self.ship.addItem(label, key)
        self.system = QComboBox()
        self.system.addItems(list(tables().systems))
        self.gravity_well = QComboBox()
        self.gravity_well.setEditable(True)
        self.region = QComboBox()
        self.region.setEditable(True)
        self.region.setMinimumWidth(140)
        self.session_button = QPushButton("Start session")
        self.prev_cluster = QPushButton("◀")
        self.next_cluster = QPushButton("▶")
        for b in (self.prev_cluster, self.next_cluster):
            b.setFixedWidth(28)
        self.cluster = QLabel("—")
        self.new_cluster = QPushButton("New rock")

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 4, 6, 4)
        for label, widget in (("Ship", self.ship), ("System", self.system),
                              ("Body", self.gravity_well), ("Region", self.region)):  # fmt: skip
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addSpacing(12)
        row.addWidget(self.session_button)
        row.addSpacing(12)
        row.addWidget(QLabel("Cluster"))
        row.addWidget(self.prev_cluster)
        row.addWidget(self.cluster)
        row.addWidget(self.next_cluster)
        row.addWidget(self.new_cluster)
        row.addStretch(1)
        self.account = QPushButton("Log in with Discord")
        self.account.setFlat(True)
        self.account.clicked.connect(self.account_clicked.emit)
        row.addWidget(self.account)

        self.ship.currentIndexChanged.connect(lambda _: self.ship_changed.emit(self.ship_key))
        self.system.currentTextChanged.connect(self._system_changed)
        self.gravity_well.currentTextChanged.connect(self._gravity_well_changed)
        self.region.currentTextChanged.connect(lambda _: self._emit_location())
        self.session_button.clicked.connect(self._session_clicked)
        self.prev_cluster.clicked.connect(lambda: self.cluster_step.emit(-1))
        self.next_cluster.clicked.connect(lambda: self.cluster_step.emit(1))
        self.new_cluster.clicked.connect(self.new_cluster_clicked.emit)
        self._active = False

    @property
    def ship_key(self) -> str:
        return str(self.ship.currentData())

    def set_ship(self, key: str) -> None:
        index = self.ship.findData(key)
        if index >= 0:
            self.ship.setCurrentIndex(index)

    def set_location(self, system: str, gravity_well: str, region: str) -> None:
        for w in (self.system, self.gravity_well, self.region):
            w.blockSignals(True)
        self.system.setCurrentText(system)
        self._fill_gravity_wells(system)
        self.gravity_well.setCurrentText(gravity_well)
        self._fill_regions(gravity_well)
        self.region.setCurrentText(region)
        for w in (self.system, self.gravity_well, self.region):
            w.blockSignals(False)

    def location(self) -> tuple[str, str, str]:
        return (
            self.system.currentText().strip(),
            self.gravity_well.currentText().strip(),
            self.region.currentText().strip(),
        )

    def set_session(self, active: bool, cluster_number: int, cluster_count: int) -> None:
        self._active = active
        self.session_button.setText("Close session" if active else "Start session")
        self.ship.setEnabled(True)
        for w in (self.prev_cluster, self.next_cluster, self.new_cluster):
            w.setEnabled(active)
        self.prev_cluster.setEnabled(active and cluster_number > 1)
        self.next_cluster.setEnabled(active and cluster_number < cluster_count)
        self.cluster.setText(f"#{cluster_number} of {cluster_count}" if active else "—")

    def _session_clicked(self) -> None:
        (self.close_clicked if self._active else self.start_clicked).emit()

    def _system_changed(self, system: str) -> None:
        self._fill_gravity_wells(system)
        self._emit_location()

    def _gravity_well_changed(self, gravity_well: str) -> None:
        self._fill_regions(gravity_well)
        self._emit_location()

    def _fill_gravity_wells(self, system: str) -> None:
        current = self.gravity_well.currentText()
        self.gravity_well.blockSignals(True)
        self.gravity_well.clear()
        self.gravity_well.addItems(["", *tables().gravity_wells_of(system)])
        self.gravity_well.setCurrentText(current)
        self.gravity_well.blockSignals(False)

    def _fill_regions(self, gravity_well: str) -> None:
        current = self.region.currentText()
        self.region.blockSignals(True)
        self.region.clear()
        self.region.addItems(["", *tables().regions_of(gravity_well)])
        self.region.setCurrentText(current)
        self.region.blockSignals(False)

    def _emit_location(self) -> None:
        self.location_changed.emit(*self.location())
