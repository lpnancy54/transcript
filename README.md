# VoiceScribe Medical (PySide6)

Application desktop de transcription audio/vidéo avec faster-whisper.

## Fonctionnalités

- Enregistrement micro (sélection du périphérique + vumètre).
- Import de fichiers audio/vidéo.
- Arborescence locale des sessions avec menu contextuel (clic droit):
  - **Transcrire**
  - **Lire**
  - **Renommer**
  - **Supprimer**
- Création de dossiers pour classer les enregistrements.
- Génération automatique d'un `*_compte_rendu.txt` avec extraction:
  - Nom / prénom
  - Date de naissance
  - Motif
  - Praticiens/intervenants mentionnés.
- Support d'un logo local si `logo.png`, `logo.jpg`, `logo.jpeg` ou `logo.ico` est présent à la racine.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Sorties

Dans chaque dossier d'enregistrement:

- `*_transcription.txt`
- `*_compte_rendu.txt`
- `*_meta.json`
