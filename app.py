from __future__ import annotations

import json
import queue
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from tkinter import filedialog, messagebox, simpledialog, StringVar
import tkinter as tk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # fallback sans drag & drop
    DND_FILES = None
    TkinterDnD = None


AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass
class SessionFolder:
    name: str
    path: Path


class AudioRecorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self.q: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self._frames = []
        self.is_recording = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def start(self):
        self._frames = []
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
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

        audio = self._frames[0]
        for frame in self._frames[1:]:
            audio = self._concat(audio, frame)

        sf.write(str(output_file), audio, self.samplerate)
        return output_file

    @staticmethod
    def _concat(a, b):
        import numpy as np

        return np.concatenate((a, b), axis=0)


class Transcriber:
    def __init__(self):
        self.model_name = None
        self.model = None

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

        segments, info = self.model.transcribe(
            str(input_path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )

        full_text = []
        json_segments = []
        for seg in segments:
            full_text.append(seg.text.strip())
            json_segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
            )

        return {
            "text": "\n".join([t for t in full_text if t]),
            "segments": json_segments,
            "language": info.language,
            "language_probability": info.language_probability,
        }


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transcripteur Speech-to-Text")
        self.root.geometry("1100x700")

        self.workspace = Path.cwd() / "sessions"
        self.workspace.mkdir(exist_ok=True)

        self.transcriber = Transcriber()
        self.recorder = AudioRecorder()

        self.current_file: Optional[Path] = None
        self.current_session: Optional[SessionFolder] = None

        self.model_var = StringVar(value="small")
        self.lang_var = StringVar(value="fr")

        self._build_ui()
        self.refresh_sessions()

    def _build_ui(self):
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        left = tk.Frame(container, bd=1, relief="solid")
        left.pack(side="left", fill="y", padx=8, pady=8)

        tk.Label(left, text="Sessions", font=("Arial", 12, "bold")).pack(padx=8, pady=8)

        self.session_list = tk.Listbox(left, width=35, height=24)
        self.session_list.pack(padx=8, pady=8, fill="y")
        self.session_list.bind("<<ListboxSelect>>", self._on_session_select)

        tk.Button(left, text="Nouvelle session", command=self.create_session).pack(fill="x", padx=8, pady=4)
        tk.Button(left, text="Renommer session", command=self.rename_session).pack(fill="x", padx=8, pady=4)
        tk.Button(left, text="Supprimer session", command=self.delete_session).pack(fill="x", padx=8, pady=4)

        right = tk.Frame(container)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        top_controls = tk.Frame(right)
        top_controls.pack(fill="x", pady=6)

        tk.Label(top_controls, text="Modèle:").pack(side="left")
        tk.OptionMenu(top_controls, self.model_var, "tiny", "base", "small", "medium", "large-v3").pack(side="left", padx=6)

        tk.Label(top_controls, text="Langue:").pack(side="left", padx=(12, 0))
        tk.Entry(top_controls, textvariable=self.lang_var, width=8).pack(side="left", padx=6)

        tk.Button(top_controls, text="Transcrire", command=self.run_transcription).pack(side="right")

        self.drop_label = tk.Label(
            right,
            text="Glissez-déposez un fichier audio/vidéo ici\nou cliquez sur 'Choisir un fichier'",
            bd=2,
            relief="groove",
            height=6,
        )
        self.drop_label.pack(fill="x", pady=10)

        if DND_FILES and hasattr(self.drop_label, "drop_target_register"):
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop_file)

        buttons = tk.Frame(right)
        buttons.pack(fill="x", pady=4)
        tk.Button(buttons, text="Choisir un fichier", command=self.pick_file).pack(side="left")
        tk.Button(buttons, text="Démarrer micro", command=self.start_recording).pack(side="left", padx=6)
        tk.Button(buttons, text="Arrêter micro", command=self.stop_recording).pack(side="left", padx=6)

        self.status = tk.Label(right, text="Prêt", anchor="w")
        self.status.pack(fill="x", pady=6)

        self.file_label = tk.Label(right, text="Aucun fichier sélectionné", fg="gray")
        self.file_label.pack(fill="x")

        tk.Label(right, text="Transcription", font=("Arial", 11, "bold")).pack(anchor="w", pady=(12, 4))
        self.output_text = tk.Text(right, wrap="word")
        self.output_text.pack(fill="both", expand=True)

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
        target = self.workspace / new_name
        session.path.rename(target)
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
        path = Path(raw)
        self._set_current_file(path)

    def _set_current_file(self, path: Path):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            messagebox.showerror("Erreur", "Format non supporté.")
            return
        self.current_file = path
        self.file_label.config(text=f"Fichier: {path}", fg="black")
        self._set_status("Fichier prêt.")

    def start_recording(self):
        try:
            self.recorder.start()
            self._set_status("Enregistrement micro en cours...")
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

        session = self._ensure_session()
        thread = threading.Thread(target=self._transcribe_worker, args=(session, self.current_file), daemon=True)
        thread.start()

    def _transcribe_worker(self, session: SessionFolder, file_path: Path):
        self._set_status("Chargement du modèle...")
        try:
            self.transcriber.load_model(self.model_var.get())

            target_media = session.path / file_path.name
            if file_path.resolve() != target_media.resolve():
                shutil.copy2(file_path, target_media)
            else:
                target_media = file_path

            self._set_status("Transcription en cours...")
            result = self.transcriber.transcribe(target_media, language=self.lang_var.get().strip() or None)

            txt_path = session.path / "transcription.txt"
            json_path = session.path / "transcription.json"

            txt_path.write_text(result["text"], encoding="utf-8")
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", result["text"])

            self._set_status(f"Terminé. Fichiers sauvegardés dans: {session.path}")
        except Exception as exc:
            self._set_status(f"Erreur: {exc}")
            messagebox.showerror("Erreur transcription", str(exc))

    def _set_status(self, msg: str):
        self.status.config(text=msg)
        self.root.update_idletasks()


def build_root():
    if TkinterDnD:
        return TkinterDnD.Tk()
    return tk.Tk()


def main():
    root = build_root()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
