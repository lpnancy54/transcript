#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QDir, QThread, QTimer, Qt, Signal, Slot, QSize
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from faster_whisper import WhisperModel

APP_NAME = "VoiceScribe Medical"
APP_VERSION = "2.0.0"
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".wma", ".aac", ".opus"}
SUPPORTED_VIDEO = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v"}
SUPPORTED_ALL = SUPPORTED_AUDIO | SUPPORTED_VIDEO


@dataclass
class PatientInfo:
    nom: str = ""
    prenom: str = ""
    date_naissance: str = ""
    motif_consultation: str = ""
    praticiens: str = ""


class MedicalReportBuilder:
    @staticmethod
    def _search(text: str, patterns: list[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" .,:;\n")
        return ""

    @staticmethod
    def _sentences_with_keywords(text: str, keywords: list[str], max_items: int = 12) -> str:
        sentences = re.split(r"(?<=[\.!?])\s+", text.replace("\n", " "))
        selected = [s.strip() for s in sentences if any(k.lower() in s.lower() for k in keywords)]
        return "\n".join(selected[:max_items]) if selected else "À compléter manuellement."

    @staticmethod
    def extract_patient_info(transcript: str) -> PatientInfo:
        nom = MedicalReportBuilder._search(
            transcript,
            [
                r"nom\s*(?:du patient)?\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)",
                r"madame\s+([A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)",
                r"monsieur\s+([A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)",
            ],
        )
        prenom = MedicalReportBuilder._search(
            transcript,
            [r"pr[ée]nom\s*(?:du patient)?\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)"]
        )
        dob = MedicalReportBuilder._search(
            transcript,
            [r"date de naissance\s*[:\-]\s*(\d{2}[/-]\d{2}[/-]\d{4})", r"n[ée] le\s*(\d{2}[/-]\d{2}[/-]\d{4})"],
        )
        motif = MedicalReportBuilder._search(
            transcript,
            [r"motif(?: de la consultation)?\s*[:\-]\s*([^\.\n]+)"]
        )
        praticiens = MedicalReportBuilder._sentences_with_keywords(
            transcript,
            [
                "dr ",
                "docteur",
                "praticien",
                "infirmier",
                "infirmière",
                "kiné",
                "cardiologue",
                "neurologue",
                "radiologue",
                "orl",
                "pédiatre",
                "intervenant",
            ],
            max_items=8,
        )
        return PatientInfo(nom=nom, prenom=prenom, date_naissance=dob, motif_consultation=motif, praticiens=praticiens)

    @staticmethod
    def build_report_text(transcript: str, info: PatientInfo) -> str:
        discussion = MedicalReportBuilder._sentences_with_keywords(transcript, ["patient", "parent", "mère", "père", "praticien", "docteur"])
        symptomes = MedicalReportBuilder._sentences_with_keywords(transcript, ["sympt", "douleur", "fièvre", "toux", "fatigue", "anamnèse", "antécédent"])
        conduite = MedicalReportBuilder._sentences_with_keywords(transcript, ["précon", "ordonnance", "suivi", "traitement", "conduite", "recommand"])
        return f"""COMPTE-RENDU MÉDICAL

1) Identification du patient
- Nom : {info.nom or '[À compléter]'}
- Prénom : {info.prenom or '[À compléter]'}
- Date de naissance : {info.date_naissance or '[À compléter]'}

2) Motif de la consultation
{info.motif_consultation or 'À compléter manuellement.'}

3) Discussion (patient / parents / praticien)
{discussion}

4) Symptômes et anamnèse
{symptomes}

5) Explications du praticien et conduite à tenir
{conduite}

6) Praticiens / intervenants mentionnés
{info.praticiens}
"""


class AudioRecorder(QThread):
    level_changed = Signal(float)
    recording_finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, output_path: str, sample_rate: int = 16000, channels: int = 1, device: Optional[int] = None):
        super().__init__()
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        frames: list[np.ndarray] = []

        def _callback(indata, _frame_count, _time_info, status):
            if status:
                pass
            if not self._running:
                raise sd.CallbackAbort
            frames.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata**2))) if indata.size else 0.0
            self.level_changed.emit(rms)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=1024,
                device=self.device,
                callback=_callback,
            ):
                while self._running:
                    self.msleep(50)

            if frames:
                audio = np.concatenate(frames, axis=0)
                sf.write(self.output_path, audio, self.sample_rate)
                self.recording_finished.emit(self.output_path)
            else:
                self.error_occurred.emit("Aucune donnée audio capturée. Vérifiez le micro sélectionné.")
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class TranscriptionWorker(QThread):
    progress = Signal(str)
    segment_ready = Signal(str)
    finished = Signal(str, str)
    error_occurred = Signal(str)

    def __init__(self, media_path: str, model_name: str, language: str, output_dir: str):
        super().__init__()
        self.media_path = media_path
        self.model_name = model_name
        self.language = language
        self.output_dir = output_dir

    @staticmethod
    def _fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}" if h else f"{m:02d}:{s:02d}.{cs:02d}"

    def run(self):
        try:
            device = "cpu"
            compute_type = "int8"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
            except Exception:
                pass

            self.progress.emit(f"Chargement du modèle {self.model_name} ({device})…")
            model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
            self.progress.emit("Transcription en cours…")
            t0 = _time.perf_counter()
            lang = None if self.language == "auto" else self.language
            segments_gen, info = model.transcribe(self.media_path, language=lang, beam_size=5, vad_filter=True)

            lines_ts: list[str] = []
            lines_plain: list[str] = []
            for seg in segments_gen:
                line = f"[{self._fmt(seg.start)} → {self._fmt(seg.end)}]  {seg.text.strip()}"
                lines_ts.append(line)
                if seg.text.strip():
                    lines_plain.append(seg.text.strip())
                self.segment_ready.emit(line)

            full_ts = "\n".join(lines_ts)
            full_plain = "\n".join(lines_plain)
            elapsed = _time.perf_counter() - t0

            stem = Path(self.media_path).stem
            txt_path = Path(self.output_dir) / f"{stem}_transcription.txt"
            txt_path.write_text(full_ts + "\n\nTEXTE BRUT\n" + full_plain, encoding="utf-8")

            report_info = MedicalReportBuilder.extract_patient_info(full_plain)
            report_text = MedicalReportBuilder.build_report_text(full_plain, report_info)
            report_path = Path(self.output_dir) / f"{stem}_compte_rendu.txt"
            report_path.write_text(report_text, encoding="utf-8")

            meta = {
                "source": os.path.basename(self.media_path),
                "language": info.language,
                "confidence": info.language_probability,
                "elapsed": elapsed,
                "report": str(report_path),
            }
            (Path(self.output_dir) / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            self.finished.emit(full_ts, str(txt_path))
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class AudioDeviceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_device = None
        self.setWindowTitle("Sélection du microphone")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Choisissez le microphone :"))
        self.combo = QComboBox()
        self._map: dict[int, int] = {}
        idx = 0
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                self.combo.addItem(f"{dev['name']} ({int(dev['default_samplerate'])} Hz)")
                self._map[idx] = i
                idx += 1
        lay.addWidget(self.combo)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._ok)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _ok(self):
        self.selected_device = self._map.get(self.combo.currentIndex())
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.workspace = Path.cwd() / "sessions"
        self.workspace.mkdir(exist_ok=True)
        self._recorder: Optional[AudioRecorder] = None
        self._worker: Optional[TranscriptionWorker] = None
        self._selected_mic: Optional[int] = None
        self._is_recording = False
        self._rec_start: Optional[datetime] = None
        self._queue: list[tuple[str, str]] = []
        self._busy = False

        self._build_ui()
        self._build_menu()
        self._connect()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1280, 820)
        self.statusBar().showMessage("Prêt")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        tb = QToolBar("Outils")
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        self.addToolBar(tb)

        self.logo_label = QLabel("🎙")
        for c in ["logo.png", "logo.jpg", "logo.jpeg", "logo.ico"]:
            p = Path(c)
            if p.exists():
                self.logo_label.setText(f"Logo: {p.name}")
                break
        tb.addWidget(self.logo_label)

        self.btn_rec = QPushButton("🎙 Enregistrer")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_import = QPushButton("📂 Importer")
        self.btn_folder = QPushButton("📁 Nouveau dossier")
        self.btn_settings = QPushButton("🎤 Choisir micro")
        self.btn_transcribe = QPushButton("▶ Transcrire la sélection")

        for b in [self.btn_rec, self.btn_pause, self.btn_import, self.btn_folder, self.btn_settings]:
            tb.addWidget(b)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(self.btn_transcribe)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        l = QVBoxLayout(left)
        l.addWidget(QLabel("📁 Gestionnaire de fichiers"))
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(str(self.workspace))
        self.fs_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(str(self.workspace)))
        self.tree.hideColumn(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        l.addWidget(self.tree)

        row = QHBoxLayout()
        self.btn_rename = QPushButton("✏️ Renommer")
        self.btn_delete = QPushButton("🗑 Supprimer")
        self.btn_open = QPushButton("📂 Ouvrir dossier")
        row.addWidget(self.btn_rename)
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_open)
        l.addLayout(row)
        splitter.addWidget(left)

        right = QWidget()
        r = QVBoxLayout(right)
        self.lbl_status = QLabel("")
        self.level = QProgressBar()
        self.level.setMaximum(100)
        self.chrono = QLabel("00:00")
        top = QHBoxLayout()
        top.addWidget(QLabel("Niveau"))
        top.addWidget(self.level, 1)
        top.addWidget(self.chrono)
        r.addLayout(top)
        r.addWidget(self.lbl_status)

        tabs = QTabWidget()
        self.txt_result = QPlainTextEdit()
        self.txt_queue = QPlainTextEdit()
        self.txt_queue.setReadOnly(True)
        self.txt_result.setReadOnly(True)
        tabs.addTab(self.txt_result, "Transcription")
        tabs.addTab(self.txt_queue, "File d'attente")
        r.addWidget(tabs, 1)
        splitter.addWidget(right)

        self.timer = QTimer()
        self.timer.setInterval(500)

    def _build_menu(self):
        m = self.menuBar()
        f = m.addMenu("Fichier")
        f.addAction(QAction("Importer", self, triggered=self._import_files))
        f.addAction(QAction("Nouveau dossier", self, triggered=self._create_folder))

    def _connect(self):
        self.btn_import.clicked.connect(self._import_files)
        self.btn_folder.clicked.connect(self._create_folder)
        self.btn_rename.clicked.connect(self._rename_selected)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_open.clicked.connect(lambda: self._open_path(str(self.workspace)))
        self.btn_transcribe.clicked.connect(self._transcribe_selected)
        self.btn_rec.clicked.connect(self._toggle_record)
        self.btn_settings.clicked.connect(self._select_mic)
        self.tree.customContextMenuRequested.connect(self._tree_context)
        self.tree.doubleClicked.connect(self._on_dbl)
        self.timer.timeout.connect(self._tick)

    def _make_folder(self) -> str:
        return f"Enregistrement du {datetime.now():%Y-%m-%d %Hh%Mm%Ss}"

    def _toggle_record(self):
        if self._is_recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        folder = self.workspace / self._make_folder()
        folder.mkdir(exist_ok=True)
        wav = folder / "enregistrement.wav"
        self._recorder = AudioRecorder(str(wav), device=self._selected_mic)
        self._recorder.level_changed.connect(lambda x: self.level.setValue(min(int(x * 400), 100)))
        self._recorder.recording_finished.connect(self._on_record_done)
        self._recorder.error_occurred.connect(lambda e: QMessageBox.critical(self, "Micro", e))
        self._recorder.start()
        self._is_recording = True
        self._rec_start = datetime.now()
        self.timer.start()
        self.btn_rec.setText("⏹ Arrêter")
        self.statusBar().showMessage("Enregistrement en cours")

    def _stop_record(self):
        if self._recorder:
            self._recorder.stop()
            self._recorder.wait(5000)
        self._is_recording = False
        self.btn_rec.setText("🎙 Enregistrer")
        self.timer.stop()

    @Slot(str)
    def _on_record_done(self, path: str):
        self.statusBar().showMessage(f"Audio sauvegardé : {path}")
        self._enqueue(path, str(Path(path).parent))

    def _tick(self):
        if not self._rec_start:
            return
        d = int((datetime.now() - self._rec_start).total_seconds())
        m, s = divmod(d, 60)
        h, m = divmod(m, 60)
        self.chrono.setText(f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")

    def _select_mic(self):
        dlg = AudioDeviceDialog(self)
        if dlg.exec() and dlg.selected_device is not None:
            self._selected_mic = dlg.selected_device
            dev = sd.query_devices(dlg.selected_device)
            self.statusBar().showMessage(f"Micro sélectionné : {dev['name']}")

    def _import_files(self):
        ext_str = " ".join(f"*{e}" for e in sorted(SUPPORTED_ALL))
        paths, _ = QFileDialog.getOpenFileNames(self, "Importer", "", f"Médias ({ext_str});;Tous (*)")
        for i, src in enumerate(paths):
            folder = self.workspace / (self._make_folder() + (f" ({i+1})" if i else ""))
            folder.mkdir(exist_ok=True)
            dest = folder / Path(src).name
            shutil.copy2(src, dest)
            self._enqueue(str(dest), str(folder))

    def _enqueue(self, media_path: str, out_dir: str):
        self._queue.append((media_path, out_dir))
        self._refresh_queue()
        if not self._busy:
            self._run_next()

    def _refresh_queue(self):
        if not self._queue:
            self.txt_queue.setPlainText("File d'attente vide")
        else:
            self.txt_queue.setPlainText("\n".join(f"{i+1}. {Path(p).name}" for i, (p, _) in enumerate(self._queue)))

    def _run_next(self):
        if not self._queue:
            self._busy = False
            self.lbl_status.setText("✅ Terminé")
            return
        self._busy = True
        media_path, out_dir = self._queue.pop(0)
        self._refresh_queue()
        self.txt_result.clear()
        self._worker = TranscriptionWorker(media_path, "base", "fr", out_dir)
        self._worker.progress.connect(lambda m: self.lbl_status.setText(m))
        self._worker.segment_ready.connect(self.txt_result.appendPlainText)
        self._worker.finished.connect(self._on_done)
        self._worker.error_occurred.connect(lambda e: QMessageBox.critical(self, "Erreur", e))
        self._worker.start()

    @Slot(str, str)
    def _on_done(self, _full_text: str, txt_path: str):
        self.statusBar().showMessage(f"Transcription sauvegardée: {txt_path}")
        self._run_next()

    def _transcribe_selected(self):
        idx = self.tree.currentIndex()
        if not idx.isValid():
            QMessageBox.information(self, "Sélection", "Sélectionnez un fichier audio/vidéo.")
            return
        p = Path(self.fs_model.filePath(idx))
        if p.suffix.lower() not in SUPPORTED_ALL:
            QMessageBox.warning(self, "Format", "Format non supporté.")
            return
        self._enqueue(str(p), str(p.parent))

    def _create_folder(self):
        name, ok = QInputDialog.getText(self, "Nouveau dossier", "Nom du dossier :", text=self._make_folder())
        if ok and name.strip():
            (self.workspace / name.strip()).mkdir(exist_ok=True)

    def _rename_selected(self):
        idx = self.tree.currentIndex()
        if not idx.isValid():
            return
        path = Path(self.fs_model.filePath(idx))
        new, ok = QInputDialog.getText(self, "Renommer", "Nouveau nom :", text=path.name)
        if ok and new.strip() and new.strip() != path.name:
            path.rename(path.parent / new.strip())

    def _delete_selected(self):
        idx = self.tree.currentIndex()
        if not idx.isValid():
            return
        path = Path(self.fs_model.filePath(idx))
        if QMessageBox.question(self, "Supprimer", f"Supprimer {path.name} ?") == QMessageBox.Yes:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _tree_context(self, pos):
        idx = self.tree.indexAt(pos)
        menu = QMenu(self)
        if idx.isValid():
            path = Path(self.fs_model.filePath(idx))
            if path.suffix.lower() in SUPPORTED_ALL:
                menu.addAction("▶ Transcrire", lambda: self._enqueue(str(path), str(path.parent)))
                menu.addAction("🎧 Lire", lambda: self._open_path(str(path)))
                menu.addSeparator()
            menu.addAction("✏️ Renommer", self._rename_selected)
            menu.addAction("🗑 Supprimer", self._delete_selected)
        else:
            menu.addAction("📁 Nouveau dossier", self._create_folder)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_dbl(self, idx):
        path = Path(self.fs_model.filePath(idx))
        if path.suffix.lower() in SUPPORTED_ALL:
            self._open_path(str(path))

    @staticmethod
    def _open_path(path: str):
        QDesktopServices.openUrl(Path(path).as_uri())


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(42, 42, 46))
    pal.setColor(QPalette.WindowText, QColor(218, 218, 218))
    pal.setColor(QPalette.Base, QColor(28, 28, 32))
    pal.setColor(QPalette.Text, QColor(218, 218, 218))
    pal.setColor(QPalette.Button, QColor(52, 52, 56))
    pal.setColor(QPalette.ButtonText, QColor(218, 218, 218))
    pal.setColor(QPalette.Highlight, QColor(42, 130, 218))
    pal.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(pal)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
