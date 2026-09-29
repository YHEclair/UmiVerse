from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from umiverse_desktop import APP_DISPLAY_VERSION, APP_NAME
from umiverse_desktop.services.docking import VinaDockingService
from umiverse_desktop.services.explain import UmamiExplainer
from umiverse_desktop.services.io_utils import write_csv, write_fasta
from umiverse_desktop.services.pipeline import UmiVersePipeline
from umiverse_desktop.ui.table_model import DataFrameTableModel
from umiverse_desktop.ui.translations import TRANSLATIONS
from umiverse_desktop.ui.workers import Worker


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.pipeline = UmiVersePipeline(project_root)
        self.explainer = UmamiExplainer(project_root)
        self.docking = VinaDockingService()
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers: set[Worker] = set()
        self.language = "en"
        self.current_frame = pd.DataFrame()
        self.table_model = DataFrameTableModel()

        self.sequence_input = QLineEdit()
        self.single_table = self._table(self.table_model)
        self.batch_table_model = DataFrameTableModel()
        self.batch_table = self._table(self.batch_table_model)
        self.generated_table_model = DataFrameTableModel()
        self.generated_table = self._table(self.generated_table_model)
        self.screen_table_model = DataFrameTableModel()
        self.screen_table = self._table(self.screen_table_model)
        self.docking_table_model = DataFrameTableModel()
        self.docking_table = self._table(self.docking_table_model)
        self.residue_table_model = DataFrameTableModel()
        self.motif_table_model = DataFrameTableModel()
        self.residue_table = self._table(self.residue_table_model)
        self.motif_table = self._table(self.motif_table_model)
        self.explain_sequence_input = QLineEdit()
        self.explain_strategy_combo = QComboBox()
        self.explain_strategy_label = QLabel()
        self.explain_method_note = QLabel()
        self.explain_summary = QTextEdit()
        self.explain_summary_label = QLabel()
        self.residue_contribution_label = QLabel()
        self.motif_contribution_label = QLabel()
        self.residue_strip = QHBoxLayout()
        self.vina_path_input = QLineEdit()
        self.receptor_input = QLineEdit()
        self.ligand_dir_input = QLineEdit()
        self.config_input = QLineEdit()
        self.output_dir_input = QLineEdit()
        self.docking_cpu_spin = self._spin(1, 128, 4)
        self.docking_exhaustiveness_spin = self._spin(1, 1024, 8)
        self.pocket_params_label = QLabel()
        self.pocket_params_view = QTextEdit()

        self._build_ui()
        self._apply_language()
        self.nav.setCurrentRow(0)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(root)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(230)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(14, 14, 14, 14)

        self.brand_label = QLabel(APP_NAME)
        self.brand_label.setObjectName("brand")
        sidebar_layout.addWidget(self.brand_label)

        self.nav = QListWidget()
        self.nav.setFrameShape(QFrame.NoFrame)
        self.nav.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.nav, 1)

        self.language_button = QPushButton()
        self.language_button.clicked.connect(self._toggle_language)
        sidebar_layout.addWidget(self.language_button)
        outer.addWidget(self.sidebar)

        content = QVBoxLayout()
        self.header = QLabel()
        self.header.setObjectName("header")
        content.addWidget(self.header)

        self.pages = QStackedWidget()
        content.addWidget(self.pages, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel()
        footer.addWidget(self.status_label, 1)
        self.export_csv_button = QPushButton()
        self.export_csv_button.clicked.connect(self._export_csv)
        self.export_fasta_button = QPushButton()
        self.export_fasta_button.clicked.connect(self._export_fasta)
        footer.addWidget(self.export_csv_button)
        footer.addWidget(self.export_fasta_button)
        content.addLayout(footer)
        outer.addLayout(content, 1)

        self.pages.addWidget(self._single_page())
        self.pages.addWidget(self._batch_page())
        self.pages.addWidget(self._generate_page())
        self.pages.addWidget(self._screen_page())
        self.pages.addWidget(self._explain_page())
        self.pages.addWidget(self._docking_page())
        self.nav.setCurrentRow(0)

        self.setStyleSheet(
            """
            QMainWindow { background: #f6f7f9; }
            #sidebar { background: #20242b; }
            #brand { color: white; font-size: 24px; font-weight: 700; padding: 8px 4px 18px 4px; }
            QListWidget { background: #20242b; color: #d8dde7; font-size: 14px; }
            QListWidget::item { padding: 11px 8px; border-radius: 6px; }
            QListWidget::item:selected { background: #3c7f72; color: white; }
            #header { font-size: 20px; font-weight: 700; padding: 18px 18px 6px 18px; color: #1f2933; }
            QPushButton { padding: 8px 12px; border-radius: 6px; background: #2f6f64; color: white; }
            QPushButton:disabled { background: #aab4be; }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 6px; border: 1px solid #c7d0d9; border-radius: 4px; background: white; color: #111827;
            }
            QComboBox QAbstractItemView {
                background: white;
                color: #111827;
                selection-background-color: #d7eee9;
                selection-color: #111827;
                border: 1px solid #c7d0d9;
            }
            QCheckBox {
                color: #111827;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #8a97a5;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #2f6f64;
                border: 1px solid #2f6f64;
            }
            QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
                color: #4b5563; background: #eef2f6;
            }
            QTableView {
                background: white;
                color: #111827;
                alternate-background-color: #f8fafc;
                border: 1px solid #c7d0d9;
                gridline-color: #e2e8f0;
                selection-background-color: #d7eee9;
                selection-color: #111827;
            }
            QHeaderView::section {
                background: #eef2f6;
                color: #111827;
                padding: 6px;
                border: 1px solid #c7d0d9;
            }
            QScrollBar:horizontal {
                background: #e5ebf1;
                height: 14px;
                margin: 0px 14px 0px 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal {
                background: #7b8a99;
                min-width: 42px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #5f6f7f;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                background: #c9d4df;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
            }
            QScrollBar:vertical {
                background: #e5ebf1;
                width: 14px;
                margin: 14px 0px 14px 0px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #7b8a99;
                min-height: 42px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5f6f7f;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                background: #c9d4df;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QMessageBox {
                background: white;
                color: #111827;
            }
            QMessageBox QLabel {
                color: #111827;
                background: white;
            }
            QMessageBox QTextEdit {
                background: white;
                color: #111827;
                border: 1px solid #c7d0d9;
            }
            QMessageBox QPushButton {
                background: #2f6f64;
                color: white;
                min-width: 72px;
            }
            QLabel { color: #1f2933; }
            """
        )

    def _single_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QHBoxLayout()
        self.single_sequence_label = QLabel()
        form.addWidget(self.single_sequence_label)
        form.addWidget(self.sequence_input, 1)
        self.single_run_button = QPushButton()
        self.single_run_button.clicked.connect(self._run_single)
        form.addWidget(self.single_run_button)
        layout.addLayout(form)
        layout.addWidget(self.single_table, 1)
        return page

    def _batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.load_file_button = QPushButton()
        self.load_file_button.clicked.connect(self._load_batch_file)
        controls.addWidget(self.load_file_button)
        self.include_umami = QCheckBox()
        self.include_umami.setChecked(True)
        controls.addWidget(self.include_umami)
        self.include_toxicity = QCheckBox()
        self.include_toxicity.setChecked(True)
        controls.addWidget(self.include_toxicity)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self.batch_table, 1)
        return page

    def _generate_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(self._generation_controls(prefix="gen"))
        self.generate_run_button = QPushButton()
        self.generate_run_button.clicked.connect(self._run_generate)
        layout.addWidget(self.generate_run_button, alignment=Qt.AlignLeft)
        layout.addWidget(self.generated_table, 1)
        return page

    def _screen_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(self._generation_controls(prefix="screen"))
        thresholds = QFormLayout()
        self.min_umami_spin = self._double_spin(0.0, 1.0, 0.8, 0.05)
        self.max_toxic_spin = self._double_spin(0.0, 1.0, 0.2, 0.05)
        self.min_umami_label = QLabel()
        self.max_toxic_label = QLabel()
        thresholds.addRow(self.min_umami_label, self.min_umami_spin)
        thresholds.addRow(self.max_toxic_label, self.max_toxic_spin)
        layout.addLayout(thresholds)
        self.screen_run_button = QPushButton()
        self.screen_run_button.clicked.connect(self._run_screen)
        layout.addWidget(self.screen_run_button, alignment=Qt.AlignLeft)
        layout.addWidget(self.screen_table, 1)
        return page

    def _generation_controls(self, prefix: str) -> QHBoxLayout:
        controls = QHBoxLayout()
        mode = QComboBox()
        for value in ("fixed", "range", "empirical"):
            mode.addItem(value, value)
        mode.currentIndexChanged.connect(
            lambda _index, p=prefix: self._update_generation_mode_visibility(p)
        )
        count = self._spin(1, 100000, 100)
        fixed = self._spin(2, 20, 8)
        min_len = self._spin(2, 20, 2)
        max_len = self._spin(2, 20, 20)
        steps = self._spin(10, 1000, 200)
        temp = self._double_spin(0.1, 3.0, 1.0, 0.1)
        top_p = self._double_spin(0.1, 1.0, 0.95, 0.05)
        widgets = {
            "mode": mode,
            "count": count,
            "fixed": fixed,
            "min": min_len,
            "max": max_len,
            "steps": steps,
            "temp": temp,
            "top_p": top_p,
        }
        pairs = {}
        setattr(self, f"{prefix}_widgets", widgets)
        for key, widget in widgets.items():
            label = QLabel()
            setattr(self, f"{prefix}_{key}_label", label)
            pairs[key] = (label, widget)
            controls.addWidget(label)
            controls.addWidget(widget)
        setattr(self, f"{prefix}_pairs", pairs)
        controls.addStretch(1)
        self._update_generation_mode_visibility(prefix)
        return controls

    def _explain_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.explain_sequence_label = QLabel()
        row.addWidget(self.explain_sequence_label)
        row.addWidget(self.explain_sequence_input, 1)
        row.addWidget(self.explain_strategy_label)
        for value in ("ag_replacement", "alanine_scanning", "average_19aa"):
            self.explain_strategy_combo.addItem(value, value)
        row.addWidget(self.explain_strategy_combo)
        self.explain_run_button = QPushButton()
        self.explain_run_button.clicked.connect(self._run_explain)
        row.addWidget(self.explain_run_button)
        layout.addLayout(row)
        self.explain_method_note.setWordWrap(True)
        layout.addWidget(self.explain_method_note)
        self.explain_summary.setReadOnly(True)
        self.explain_summary.setFixedHeight(88)
        layout.addWidget(self.explain_summary_label)
        layout.addWidget(self.explain_summary)
        strip_box = QFrame()
        strip_box.setLayout(self.residue_strip)
        layout.addWidget(strip_box)
        layout.addWidget(self.residue_contribution_label)
        layout.addWidget(self.residue_table, 1)
        layout.addWidget(self.motif_contribution_label)
        layout.addWidget(self.motif_table, 1)
        return page

    def _docking_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.vina_path_label = QLabel()
        self.receptor_label = QLabel()
        self.ligand_dir_label = QLabel()
        self.config_label = QLabel()
        self.output_dir_label = QLabel()
        self.docking_cpu_label = QLabel()
        self.docking_exhaustiveness_label = QLabel()
        form.addRow(self.vina_path_label, self._path_row(self.vina_path_input, self._browse_vina_path))
        form.addRow(self.receptor_label, self._path_row(self.receptor_input, self._browse_receptor))
        form.addRow(self.ligand_dir_label, self._path_row(self.ligand_dir_input, self._browse_ligand_dir))
        form.addRow(self.config_label, self._path_row(self.config_input, self._browse_config))
        form.addRow(self.output_dir_label, self._path_row(self.output_dir_input, self._browse_output_dir))
        form.addRow(self.docking_cpu_label, self.docking_cpu_spin)
        form.addRow(self.docking_exhaustiveness_label, self.docking_exhaustiveness_spin)
        layout.addLayout(form)
        self.pocket_params_view.setReadOnly(True)
        self.pocket_params_view.setFixedHeight(92)
        layout.addWidget(self.pocket_params_label)
        layout.addWidget(self.pocket_params_view)
        self.docking_run_button = QPushButton("Run docking")
        self.docking_run_button.clicked.connect(self._run_docking)
        layout.addWidget(self.docking_run_button, alignment=Qt.AlignLeft)
        layout.addWidget(self.docking_table, 1)
        return page

    def _path_row(self, line_edit: QLineEdit, browse_callback) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton("Browse")
        button.clicked.connect(browse_callback)
        layout.addWidget(button)
        return widget

    def _table(self, model: DataFrameTableModel) -> QTableView:
        table = QTableView()
        table.setModel(model)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(False)
        return table

    def _fit_table_columns(self, table: QTableView, model: DataFrameTableModel) -> None:
        widths = model.column_widths(table.fontMetrics())
        for column, width in enumerate(widths):
            table.setColumnWidth(column, width)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        return spin

    def _apply_language(self) -> None:
        t = TRANSLATIONS[self.language]
        app_title = self._versioned_title(t["app_title"])
        self.setWindowTitle(app_title)
        self.header.setText(app_title)
        self.language_button.setText(t["language"])
        labels = ["single", "batch", "generate", "screen", "explain", "docking"]
        self.nav.clear()
        for key in labels:
            self.nav.addItem(QListWidgetItem(t[key]))
        self.export_csv_button.setText(t["export_csv"])
        self.export_fasta_button.setText(t["export_fasta"])
        self.single_sequence_label.setText(t["sequence"])
        self.explain_sequence_label.setText(t["sequence"])
        self.explain_strategy_label.setText(t["explain_strategy"])
        current_strategy = self.explain_strategy_combo.currentData() or "ag_replacement"
        self.explain_strategy_combo.blockSignals(True)
        self.explain_strategy_combo.clear()
        for value in ("ag_replacement", "alanine_scanning", "average_19aa"):
            self.explain_strategy_combo.addItem(t[value], value)
        strategy_index = self.explain_strategy_combo.findData(current_strategy)
        self.explain_strategy_combo.setCurrentIndex(max(0, strategy_index))
        self.explain_strategy_combo.blockSignals(False)
        self.explain_method_note.setText(t["explain_method_note"])
        self.explain_summary_label.setText(t["explain_summary"])
        self.residue_contribution_label.setText(t["residue_contribution"])
        self.motif_contribution_label.setText(t["motif_contribution"])
        self.vina_path_label.setText(t["vina_executable"])
        self.receptor_label.setText(t["receptor_pdbqt"])
        self.ligand_dir_label.setText(t["ligand_pdbqt_folder"])
        self.config_label.setText(t["config_file"])
        self.output_dir_label.setText(t["output_folder"])
        self.docking_cpu_label.setText(t["docking_cpu"])
        self.docking_exhaustiveness_label.setText(t["docking_exhaustiveness"])
        self.pocket_params_label.setText(t["pocket_parameters"])
        self.docking_run_button.setText(t["run_docking"])
        self.single_run_button.setText(t["run"])
        self.explain_run_button.setText(t["run"])
        self.load_file_button.setText(t["load_file"])
        self.include_umami.setText(t["include_umami"])
        self.include_toxicity.setText(t["include_toxicity"])
        self.generate_run_button.setText(t["run"])
        self.screen_run_button.setText(t["run"])
        self.min_umami_label.setText(t["min_umami"])
        self.max_toxic_label.setText(t["max_toxic"])
        self.status_label.setText(t["status_ready"])
        for prefix in ("gen", "screen"):
            widgets = getattr(self, f"{prefix}_widgets")
            current_mode = widgets["mode"].currentData() or "fixed"
            widgets["mode"].blockSignals(True)
            widgets["mode"].clear()
            for value in ("fixed", "range", "empirical"):
                widgets["mode"].addItem(t[value], value)
            index = widgets["mode"].findData(current_mode)
            widgets["mode"].setCurrentIndex(max(0, index))
            widgets["mode"].blockSignals(False)
            mapping = {
                "mode": "mode",
                "count": "count",
                "fixed": "fixed_length",
                "min": "min_length",
                "max": "max_length",
                "steps": "sampling_steps",
                "temp": "temperature",
                "top_p": "top_p",
            }
            for key, text_key in mapping.items():
                getattr(self, f"{prefix}_{key}_label").setText(t[text_key])
            self._update_generation_mode_visibility(prefix)

    @staticmethod
    def _versioned_title(title: str) -> str:
        return title.replace(APP_NAME, f"{APP_NAME} {APP_DISPLAY_VERSION}", 1)

    def _toggle_language(self) -> None:
        current_row = self.nav.currentRow()
        self.language = "en" if self.language == "zh" else "zh"
        self._apply_language()
        self.nav.setCurrentRow(max(0, current_row))

    def _switch_page(self, row: int) -> None:
        if row >= 0:
            self.pages.setCurrentIndex(row)

    def _run_worker(self, function, on_success, *args, **kwargs) -> None:
        self.status_label.setText(TRANSLATIONS[self.language]["status_running"])
        self._set_buttons_enabled(False)
        worker = Worker(function, *args, **kwargs)
        self.active_workers.add(worker)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._worker_success(
                result,
                on_success,
                active_worker,
            )
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker: self._worker_failed(message, active_worker)
        )
        self.thread_pool.start(worker)

    def _worker_success(self, result, on_success, worker: Worker) -> None:
        self.active_workers.discard(worker)
        self._set_buttons_enabled(True)
        on_success(result)
        self.status_label.setText(TRANSLATIONS[self.language]["status_done"])

    def _worker_failed(self, message: str, worker: Worker) -> None:
        self.active_workers.discard(worker)
        self._set_buttons_enabled(True)
        self.status_label.setText("Failed.")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("UmiVerse")
        box.setText("Task failed")
        box.setDetailedText(message)
        box.setStyleSheet(
            """
            QMessageBox { background: white; color: #111827; }
            QLabel { color: #111827; background: white; }
            QTextEdit { background: white; color: #111827; border: 1px solid #c7d0d9; }
            QPushButton { background: #2f6f64; color: white; padding: 6px 12px; }
            """
        )
        box.exec()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            self.single_run_button,
            self.load_file_button,
            self.generate_run_button,
            self.screen_run_button,
            self.explain_run_button,
            self.docking_run_button,
            self.export_csv_button,
            self.export_fasta_button,
        ):
            button.setEnabled(enabled)

    def _run_single(self) -> None:
        self._run_worker(self.pipeline.predict_single, self._show_single_result, self.sequence_input.text())

    def _show_single_result(self, frame: pd.DataFrame) -> None:
        self.current_frame = frame
        self.table_model.set_frame(frame)
        self._fit_table_columns(self.single_table, self.table_model)

    def _load_batch_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open FASTA/CSV",
            str(self.project_root),
            "Sequence files (*.csv *.fa *.fasta *.faa *.fas);;All files (*)",
        )
        if not path:
            return
        self._run_worker(
            self.pipeline.predict_file,
            self._show_batch_result,
            Path(path),
            self.include_umami.isChecked(),
            self.include_toxicity.isChecked(),
        )

    def _show_batch_result(self, frame: pd.DataFrame) -> None:
        self.current_frame = frame
        self.batch_table_model.set_frame(frame)
        self._fit_table_columns(self.batch_table, self.batch_table_model)

    def _generation_params(self, prefix: str) -> dict:
        widgets = getattr(self, f"{prefix}_widgets")
        return {
            "mode": widgets["mode"].currentData(),
            "count": widgets["count"].value(),
            "fixed_length": widgets["fixed"].value(),
            "min_length": widgets["min"].value(),
            "max_length": widgets["max"].value(),
            "sampling_steps": widgets["steps"].value(),
            "temperature": widgets["temp"].value(),
            "top_p": widgets["top_p"].value(),
        }

    def _update_generation_mode_visibility(self, prefix: str) -> None:
        if not hasattr(self, f"{prefix}_widgets") or not hasattr(self, f"{prefix}_pairs"):
            return
        widgets = getattr(self, f"{prefix}_widgets")
        pairs = getattr(self, f"{prefix}_pairs")
        mode = widgets["mode"].currentData() or "fixed"
        visible_keys = {
            "fixed": {"mode", "count", "fixed", "steps", "temp", "top_p"},
            "range": {"mode", "count", "min", "max", "steps", "temp", "top_p"},
            "empirical": {"mode", "count", "min", "max", "steps", "temp", "top_p"},
        }.get(mode, {"mode", "count", "fixed", "steps", "temp", "top_p"})
        for key, (label, widget) in pairs.items():
            visible = key in visible_keys
            label.setVisible(visible)
            widget.setVisible(visible)

    def _run_generate(self) -> None:
        self._run_worker(self.pipeline.generate_candidates, self._show_generated_result, **self._generation_params("gen"))

    def _show_generated_result(self, frame: pd.DataFrame) -> None:
        self.current_frame = frame
        self.generated_table_model.set_frame(frame)
        self._fit_table_columns(self.generated_table, self.generated_table_model)

    def _run_screen(self) -> None:
        params = self._generation_params("screen")
        params["min_umami"] = self.min_umami_spin.value()
        params["max_toxic"] = self.max_toxic_spin.value()
        self._run_worker(self.pipeline.screen_generated, self._show_screen_result, **params)

    def _show_screen_result(self, frame: pd.DataFrame) -> None:
        self.current_frame = frame
        self.screen_table_model.set_frame(frame)
        self._fit_table_columns(self.screen_table, self.screen_table_model)

    def _run_explain(self) -> None:
        self._run_worker(
            self.explainer.explain_sequence,
            self._show_explain_result,
            self.explain_sequence_input.text(),
            self.explain_strategy_combo.currentData(),
        )

    def _show_explain_result(self, result: dict) -> None:
        residue_df = result["residue"]
        motif_df = result["motif"]
        self.current_frame = residue_df
        self.explain_summary.setPlainText(result["summary"])
        self.residue_table_model.set_frame(residue_df)
        self.motif_table_model.set_frame(motif_df.head(100))
        self._fit_table_columns(self.residue_table, self.residue_table_model)
        self._fit_table_columns(self.motif_table, self.motif_table_model)
        self._render_residue_strip(residue_df)

    def _browse_vina_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Vina executable", str(self.project_root), "Executable files (*.exe *);;All files (*)")
        if path:
            self.vina_path_input.setText(path)

    def _browse_receptor(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select receptor PDBQT", str(self.project_root), "PDBQT (*.pdbqt);;All files (*)")
        if path:
            self.receptor_input.setText(path)

    def _browse_ligand_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select ligand PDBQT folder", str(self.project_root))
        if path:
            self.ligand_dir_input.setText(path)

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Vina config", str(self.project_root), "Config files (*.txt *.conf *.cfg);;All files (*)")
        if path:
            self.config_input.setText(path)
            self._update_pocket_params_view(Path(path))

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", str(self.project_root))
        if path:
            self.output_dir_input.setText(path)

    def _run_docking(self) -> None:
        self._update_pocket_params_view(Path(self.config_input.text().strip()))
        cpu = self.docking_cpu_spin.value()
        exhaustiveness = self.docking_exhaustiveness_spin.value()
        self._run_worker(
            self.docking.run_batch,
            self._show_docking_result,
            self.vina_path_input.text().strip(),
            self.receptor_input.text().strip(),
            self.ligand_dir_input.text().strip(),
            self.config_input.text().strip(),
            self.output_dir_input.text().strip(),
            cpu,
            exhaustiveness,
        )

    def _show_docking_result(self, frame: pd.DataFrame) -> None:
        self.current_frame = frame
        self.docking_table_model.set_frame(frame)
        self._fit_table_columns(self.docking_table, self.docking_table_model)

    def _update_pocket_params_view(self, config_path: Path) -> None:
        try:
            pocket = self.docking.parse_pocket_config(config_path)
            self.pocket_params_view.setPlainText(
                "\n".join(f"{key} = {pocket[key]}" for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"))
            )
        except Exception as exc:
            self.pocket_params_view.setPlainText(f"Could not read pocket parameters: {exc}")

    def _render_residue_strip(self, residue_df: pd.DataFrame) -> None:
        while self.residue_strip.count():
            item = self.residue_strip.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if residue_df.empty:
            return
        score_column = "residue_importance_score"
        max_delta = max(float(residue_df[score_column].abs().max()), 1e-9)
        ordered = residue_df.sort_values("position")
        for _, row in ordered.iterrows():
            score = float(row[score_column])
            intensity = min(1.0, max(0.0, score / max_delta))
            red = int(245 - 70 * intensity)
            green = int(248 - 120 * intensity)
            blue = int(250 - 160 * intensity)
            label = QLabel(f"{row['original_residue']}\n{score:.3f}")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(58, 48)
            label.setStyleSheet(
                f"background: rgb({red},{green},{blue}); border: 1px solid #9aa6b2; border-radius: 4px;"
            )
            self.residue_strip.addWidget(label)
        self.residue_strip.addStretch(1)

    def _export_csv(self) -> None:
        if self.current_frame.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", str(self.project_root / "outputs.csv"), "CSV (*.csv)")
        if path:
            write_csv(self.current_frame, Path(path))

    def _export_fasta(self) -> None:
        if self.current_frame.empty or "sequence" not in self.current_frame.columns:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export FASTA",
            str(self.project_root / "outputs.fasta"),
            "FASTA (*.fasta *.fa)",
        )
        if path:
            write_fasta(self.current_frame, Path(path))
