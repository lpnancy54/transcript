from __future__ import annotations

import json
import queue
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from docx import Document
from faster_whisper import WhisperModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from tkinter import filedialog, messagebox, simpledialog, StringVar
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass
class SessionFolder:
    name: str
    path: Path


@dataclass
class PatientInfo:
    nom: str = ""
    prenom: str = ""
    date_naissance: str = ""
    motif_consultation: str = ""


class AudioRecorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self.device: Optional[int] = None
        self.q: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self._frames = []
        self.is_recording = False
        self.level = 0.0

    def _callback(self, indata, _frames, _time, status):
        if status:
            print(status)
        self.q.put(indata.copy())
        rms = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
        self.level = min(max(rms * 30.0, 0.0), 1.0)

    def start(self, device: Optional[int]):
        self._frames = []
        self.device = device
        self.is_recording = True
        self.level = 0.0
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=device,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self, output_file: Path) -> Path:
        if not self.stream:
            raise RuntimeError("Aucun enregistrement en cours.")

        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        self.stream = None

        while not self.q.empty():
            self._frames.append(self.q.get())

        if not self._frames:
            raise RuntimeError("Aucun audio capturé.")

        audio = np.concatenate(self._frames, axis=0)
        sf.write(str(output_file), audio, self.samplerate)
        self.level = 0.0
        return output_file


class Transcriber:
    def __init__(self):
        self.model_name: Optional[str] = None
        self.model: Optional[WhisperModel] = None

    def load_model(self, model_name: str):
        if self.model is not None and self.model_name == model_name:
            return

        device = "cuda" if self._cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.model_name = model_name

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def transcribe(self, input_path: Path, language: Optional[str] = "fr"):
        if self.model is None:
            raise RuntimeError("Modèle non chargé.")

        segments, info = self.model.transcribe(str(input_path), language=language, vad_filter=True, beam_size=5)

        full_text = []
        json_segments = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                full_text.append(text)
            json_segments.append({"start": seg.start, "end": seg.end, "text": text})

        return {
            "text": "\n".join(full_text),
            "segments": json_segments,
            "language": info.language,
            "language_probability": info.language_probability,
        }


class MedicalReportBuilder:
    @staticmethod
    def extract_patient_info(transcript: str) -> PatientInfo:
        nom = MedicalReportBuilder._search(transcript, [r"nom\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)"])
        prenom = MedicalReportBuilder._search(transcript, [r"pr[ée]nom\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ'\- ]+)"])
        dob = MedicalReportBuilder._search(
            transcript,
            [
                r"date de naissance\s*[:\-]\s*(\d{2}[/-]\d{2}[/-]\d{4})",
                r"n[ée] le\s*(\d{2}[/-]\d{2}[/-]\d{4})",
            ],
        )
        motif = MedicalReportBuilder._search(transcript, [r"motif(?: de la consultation)?\s*[:\-]\s*([^\.\n]+)"])
        return PatientInfo(nom=nom, prenom=prenom, date_naissance=dob, motif_consultation=motif)

    @staticmethod
    def _search(text: str, patterns: list[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _sentences_with_keywords(text: str, keywords: list[str]) -> str:
        sentences = re.split(r"(?<=[\.!?])\s+", text.replace("\n", " "))
        selected = [s.strip() for s in sentences if any(k.lower() in s.lower() for k in keywords)]
        return "\n".join(selected[:10]) if selected else "À compléter manuellement."

    @staticmethod
    def build_report_text(transcript: str, info: PatientInfo) -> str:
        discussion = MedicalReportBuilder._sentences_with_keywords(
            transcript,
            ["patient", "parent", "mère", "père", "praticien", "docteur"],
        )
        symptomes = MedicalReportBuilder._sentences_with_keywords(
            transcript,
            ["sympt", "douleur", "fièvre", "toux", "fatigue", "anamnèse", "antécédent"],
        )
        conduite = MedicalReportBuilder._sentences_with_keywords(
            transcript,
            ["précon", "ordonnance", "suivi", "traitement", "conduite", "recommand"],
        )
        praticiens = MedicalReportBuilder._sentences_with_keywords(
            transcript,
            ["spécialiste", "cardiologue", "radiologue", "kiné", "ORL", "pédiatre", "neurologue"],
        )

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

6) Praticiens mentionnés dans le dossier
{praticiens}

---
Transcript source :
{transcript}
"""

    @staticmethod
    def export_docx(path: Path, report_text: str):
        doc = Document()
        for line in report_text.split("\n"):
            doc.add_paragraph(line)
        doc.save(str(path))

    @staticmethod
    def export_pdf(path: Path, report_text: str):
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 40
        for line in report_text.split("\n"):
            if y < 40:
                c.showPage()
                y = height - 40
            c.drawString(40, y, line[:140])
            y -= 14
        c.save()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transcripteur Speech-to-Text + Rapport médical")
        self.root.geometry("1250x760")

        self.workspace = Path.cwd() / "sessions"
        self.workspace.mkdir(exist_ok=True)

        self.transcriber = Transcriber()
        self.recorder = AudioRecorder()

        self.current_file: Optional[Path] = None
        self.current_session: Optional[SessionFolder] = None
        self.last_transcript: str = ""
        self.transcript_validated = False

        self.model_var = StringVar(value="small")
        self.lang_var = StringVar(value="fr")
        self.mic_var = StringVar(value="")

        self.patient_nom = StringVar(value="")
        self.patient_prenom = StringVar(value="")
        self.patient_dob = StringVar(value="")
        self.patient_motif = StringVar(value="")

        self._mic_map: dict[str, int] = {}

        self._build_ui()
        self.refresh_sessions()
        self.refresh_microphones()
        self._refresh_vumeter()

    def _build_ui(self):
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        left = tk.Frame(container, bd=1, relief="solid")
        left.pack(side="left", fill="y", padx=8, pady=8)

        tk.Label(left, text="Sessions", font=("Arial", 12, "bold")).pack(padx=8, pady=8)
        self.session_list = tk.Listbox(left, width=34, height=24)
        self.session_list.pack(padx=8, pady=8, fill="y")
        self.session_list.bind("<<ListboxSelect>>", self._on_session_select)

        tk.Button(left, text="Nouvelle session", command=self.create_session).pack(fill="x", padx=8, pady=4)
        tk.Button(left, text="Renommer session", command=self.rename_session).pack(fill="x", padx=8, pady=4)
        tk.Button(left, text="Supprimer session", command=self.delete_session).pack(fill="x", padx=8, pady=4)

        right = tk.Frame(container)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        controls = tk.Frame(right)
        controls.pack(fill="x", pady=6)

        tk.Label(controls, text="Modèle:").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(controls, self.model_var, "tiny", "base", "small", "medium", "large-v3").grid(row=0, column=1, sticky="w")

        tk.Label(controls, text="Langue:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        tk.Entry(controls, textvariable=self.lang_var, width=8).grid(row=0, column=3, sticky="w")

        tk.Label(controls, text="Micro:").grid(row=0, column=4, sticky="w", padx=(10, 0))
        self.mic_combo = ttk.Combobox(controls, textvariable=self.mic_var, width=40, state="readonly")
        self.mic_combo.grid(row=0, column=5, sticky="w")
        tk.Button(controls, text="Rafraîchir micros", command=self.refresh_microphones).grid(row=0, column=6, padx=6)

        tk.Button(controls, text="Transcrire", command=self.run_transcription).grid(row=0, column=7, padx=6)

        self.drop_label = tk.Label(
            right,
            text="Glissez-déposez un fichier audio/vidéo ici\nou cliquez sur 'Choisir un fichier'",
            bd=2,
            relief="groove",
            height=4,
        )
        self.drop_label.pack(fill="x", pady=8)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self._on_drop_file)

        row2 = tk.Frame(right)
        row2.pack(fill="x", pady=4)
        tk.Button(row2, text="Choisir un fichier", command=self.pick_file).pack(side="left")
        tk.Button(row2, text="Démarrer micro", command=self.start_recording).pack(side="left", padx=6)
        tk.Button(row2, text="Arrêter micro", command=self.stop_recording).pack(side="left", padx=6)

        tk.Label(row2, text="VU-mètre:").pack(side="left", padx=(20, 4))
        self.vumeter = ttk.Progressbar(row2, orient="horizontal", mode="determinate", length=220, maximum=100)
        self.vumeter.pack(side="left")

        self.status = tk.Label(right, text="Prêt", anchor="w")
        self.status.pack(fill="x", pady=4)
        self.file_label = tk.Label(right, text="Aucun fichier sélectionné", fg="gray")
        self.file_label.pack(fill="x")

        center = tk.PanedWindow(right, orient="horizontal", sashrelief="raised")
        center.pack(fill="both", expand=True, pady=8)

        trans_frame = tk.Frame(center)
        center.add(trans_frame, minsize=500)
        tk.Label(trans_frame, text="Transcription", font=("Arial", 11, "bold")).pack(anchor="w")
        self.output_text = tk.Text(trans_frame, wrap="word")
        self.output_text.pack(fill="both", expand=True)

        report_frame = tk.Frame(center)
        center.add(report_frame, minsize=520)
        tk.Label(report_frame, text="Validation & Rapport médical", font=("Arial", 11, "bold")).pack(anchor="w")

        form = tk.Frame(report_frame)
        form.pack(fill="x", pady=4)
        self._labeled_entry(form, "Nom patient", self.patient_nom, 0)
        self._labeled_entry(form, "Prénom patient", self.patient_prenom, 1)
        self._labeled_entry(form, "Date de naissance", self.patient_dob, 2)
        self._labeled_entry(form, "Motif consultation", self.patient_motif, 3, width=45)

        actions = tk.Frame(report_frame)
        actions.pack(fill="x", pady=6)
        self.validate_button = tk.Button(actions, text="Valider la transcription", command=self.validate_transcription, state="disabled")
        self.validate_button.pack(side="left")
        tk.Button(actions, text="Générer DOCX", command=lambda: self.generate_report("docx")).pack(side="left", padx=6)
        tk.Button(actions, text="Générer PDF", command=lambda: self.generate_report("pdf")).pack(side="left", padx=6)

        self.report_preview = tk.Text(report_frame, wrap="word")
        self.report_preview.pack(fill="both", expand=True)

    def _labeled_entry(self, parent, label: str, variable: StringVar, row: int, width: int = 28):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=2, pady=2)
        tk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="w", padx=2, pady=2)

    def refresh_microphones(self):
        self._mic_map.clear()
        names = []
        for idx, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0)) > 0:
                label = f"{idx} - {device.get('name', 'Microphone')}"
                self._mic_map[label] = idx
                names.append(label)

        self.mic_combo["values"] = names
        if names:
            if self.mic_var.get() not in names:
                self.mic_var.set(names[0])
        else:
            self.mic_var.set("")
            messagebox.showwarning("Microphone", "Aucun micro détecté.")

    def _refresh_vumeter(self):
        level_percent = int(self.recorder.level * 100)
        self.vumeter["value"] = level_percent
        self.root.after(100, self._refresh_vumeter)

    def refresh_sessions(self):
        self.session_list.delete(0, tk.END)
        for path in sorted(self.workspace.glob("*")):
            if path.is_dir():
                self.session_list.insert(tk.END, path.name)

    def _get_selected_session(self) -> Optional[SessionFolder]:
        selection = self.session_list.curselection()
        if not selection:
            return None
        name = self.session_list.get(selection[0])
        return SessionFolder(name=name, path=self.workspace / name)

    def _on_session_select(self, _event=None):
        self.current_session = self._get_selected_session()
        if self.current_session:
            self._set_status(f"Session active: {self.current_session.name}")

    def create_session(self):
        default_name = f"Enregistrement du {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
        name = simpledialog.askstring("Nouvelle session", "Nom de la session:", initialvalue=default_name)
        if not name:
            return
        path = self.workspace / name
        path.mkdir(exist_ok=True)
        self.refresh_sessions()
        self._set_status(f"Session créée: {name}")

    def rename_session(self):
        session = self._get_selected_session()
        if not session:
            messagebox.showwarning("Attention", "Sélectionnez une session.")
            return
        new_name = simpledialog.askstring("Renommer", "Nouveau nom:", initialvalue=session.name)
        if not new_name:
            return
        session.path.rename(self.workspace / new_name)
        self.refresh_sessions()
        self._set_status(f"Session renommée en: {new_name}")

    def delete_session(self):
        session = self._get_selected_session()
        if not session:
            messagebox.showwarning("Attention", "Sélectionnez une session.")
            return
        if messagebox.askyesno("Confirmation", f"Supprimer la session '{session.name}' ?"):
            shutil.rmtree(session.path)
            self.refresh_sessions()
            self._set_status("Session supprimée")

    def pick_file(self):
        file_path = filedialog.askopenfilename(
            title="Choisir un fichier",
            filetypes=[("Média", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma *.mp4 *.mov *.mkv *.avi *.webm *.m4v")],
        )
        if file_path:
            self._set_current_file(Path(file_path))

    def _on_drop_file(self, event):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        self._set_current_file(Path(raw))

    def _set_current_file(self, path: Path):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            messagebox.showerror("Erreur", "Format non supporté.")
            return
        self.current_file = path
        self.file_label.config(text=f"Fichier: {path}", fg="black")
        self._set_status("Fichier prêt.")

    def start_recording(self):
        try:
            mic_label = self.mic_var.get().strip()
            if not mic_label:
                raise RuntimeError("Sélectionnez un microphone.")
            device_index = self._mic_map.get(mic_label)
            self.recorder.start(device=device_index)
            self._set_status(f"Enregistrement micro en cours ({mic_label})...")
        except Exception as exc:
            messagebox.showerror("Erreur micro", str(exc))

    def stop_recording(self):
        try:
            session = self._ensure_session()
            out = session.path / f"micro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            self.recorder.stop(out)
            self._set_current_file(out)
            self._set_status("Enregistrement terminé.")
        except Exception as exc:
            messagebox.showerror("Erreur micro", str(exc))

    def _ensure_session(self) -> SessionFolder:
        session = self._get_selected_session()
        if session:
            self.current_session = session
            return session
        default_name = f"Enregistrement du {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
        path = self.workspace / default_name
        path.mkdir(exist_ok=True)
        self.refresh_sessions()
        self.current_session = SessionFolder(default_name, path)
        return self.current_session

    def run_transcription(self):
        if not self.current_file:
            messagebox.showwarning("Attention", "Veuillez importer un fichier ou enregistrer le micro.")
            return
        if not self.current_file.exists():
            messagebox.showerror("Erreur", "Fichier introuvable.")
            return

        self.transcript_validated = False
        self.validate_button.config(state="disabled")

        session = self._ensure_session()
        thread = threading.Thread(target=self._transcribe_worker, args=(session, self.current_file), daemon=True)
        thread.start()

    def _transcribe_worker(self, session: SessionFolder, file_path: Path):
        try:
            self.root.after(0, lambda: self._set_status("Chargement du modèle..."))
            self.transcriber.load_model(self.model_var.get())

            target_media = session.path / file_path.name
            if file_path.resolve() != target_media.resolve():
                shutil.copy2(file_path, target_media)
            else:
                target_media = file_path

            self.root.after(0, lambda: self._set_status("Transcription en cours..."))
            result = self.transcriber.transcribe(target_media, language=self.lang_var.get().strip() or None)

            (session.path / "transcription.txt").write_text(result["text"], encoding="utf-8")
            (session.path / "transcription.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            self.last_transcript = result["text"]
            self.root.after(0, lambda: self._display_transcript(result["text"], session.path))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Erreur transcription", str(exc)))
            self.root.after(0, lambda: self._set_status(f"Erreur: {exc}"))

    def _display_transcript(self, text: str, session_path: Path):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.validate_button.config(state="normal")

        info = MedicalReportBuilder.extract_patient_info(text)
        self.patient_nom.set(info.nom)
        self.patient_prenom.set(info.prenom)
        self.patient_dob.set(info.date_naissance)
        self.patient_motif.set(info.motif_consultation)

        self._set_status(f"Transcription terminée. Sauvegardé dans {session_path}")

    def validate_transcription(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Validation", "Aucune transcription à valider.")
            return

        self.transcript_validated = True
        info = PatientInfo(
            nom=self.patient_nom.get().strip(),
            prenom=self.patient_prenom.get().strip(),
            date_naissance=self.patient_dob.get().strip(),
            motif_consultation=self.patient_motif.get().strip(),
        )
        report = MedicalReportBuilder.build_report_text(text, info)
        self.report_preview.delete("1.0", tk.END)
        self.report_preview.insert("1.0", report)
        self._set_status("Transcription validée. Rapport prêt à être exporté.")

    def generate_report(self, output_format: str):
        if not self.transcript_validated:
            messagebox.showwarning("Rapport", "Validez d'abord la transcription.")
            return

        session = self._ensure_session()
        report_text = self.report_preview.get("1.0", tk.END).strip()
        if not report_text:
            messagebox.showwarning("Rapport", "Aucun contenu de rapport.")
            return

        safe_nom = (self.patient_nom.get() or "Patient").replace(" ", "_")
        safe_prenom = (self.patient_prenom.get() or "Inconnu").replace(" ", "_")
        base_name = f"Compte_rendu_{safe_nom}_{safe_prenom}"

        if output_format == "docx":
            out_path = session.path / f"{base_name}.docx"
            MedicalReportBuilder.export_docx(out_path, report_text)
        elif output_format == "pdf":
            out_path = session.path / f"{base_name}.pdf"
            MedicalReportBuilder.export_pdf(out_path, report_text)
        else:
            messagebox.showerror("Rapport", "Format non supporté.")
            return

        self._set_status(f"Rapport exporté: {out_path}")
        messagebox.showinfo("Rapport", f"Rapport généré: {out_path}")

    def _set_status(self, msg: str):
        self.status.config(text=msg)
        self.root.update_idletasks()


def build_root():
    return TkinterDnD.Tk()


def main():
    root = build_root()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
