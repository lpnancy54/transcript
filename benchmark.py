from __future__ import annotations

import argparse
import time
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel


def media_duration_seconds(path: Path) -> float:
    probe = ffmpeg.probe(str(path))
    return float(probe["format"]["duration"])


def run_benchmark(file_path: Path, model_name: str, language: str | None):
    device = "cpu"
    compute_type = "int8"

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
    except Exception:
        pass

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    start = time.perf_counter()
    segments, _info = model.transcribe(str(file_path), language=language, vad_filter=True)
    _ = [seg.text for seg in segments]
    elapsed = time.perf_counter() - start

    duration = media_duration_seconds(file_path)
    rtf = elapsed / duration if duration > 0 else float("inf")

    print(f"Fichier: {file_path}")
    print(f"Durée audio: {duration:.2f}s")
    print(f"Temps transcription: {elapsed:.2f}s")
    print(f"RTF (plus bas = plus rapide): {rtf:.3f}")
    print(f"Device: {device} ({compute_type})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark local de transcription faster-whisper")
    parser.add_argument("--file", required=True, type=Path, help="Chemin vers le fichier audio/vidéo")
    parser.add_argument("--model", default="small", help="Modèle whisper: tiny/base/small/medium/large-v3")
    parser.add_argument("--language", default="fr", help="Code langue (ex: fr, en). Mettre '' pour auto")
    args = parser.parse_args()

    language = args.language or None
    run_benchmark(args.file, args.model, language)


if __name__ == "__main__":
    main()
