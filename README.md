# Transcripteur GUI (local, prêt pour SaaS)

Application Python avec interface graphique pour :

- enregistrer depuis le microphone du PC,
- importer des fichiers audio/vidéo par glisser-déposer,
- transcrire avec **faster-whisper** (rapide et fiable),
- stocker les résultats dans des dossiers nommables (ou nom par défaut `Enregistrement du ...`),
- réorganiser les dossiers et fichiers depuis l'interface.

## Pourquoi `faster-whisper` ?

`faster-whisper` (basé sur CTranslate2) est généralement plus rapide et plus sobre en mémoire que l'implémentation Whisper d'origine, avec une bonne qualité de transcription.

Le projet inclut un script de benchmark local pour **vérifier la performance sur votre machine** (CPU/GPU), car les performances réelles dépendent du matériel.

## Prérequis

- Python 3.10+
- FFmpeg installé et disponible dans le PATH

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
python app.py
```

## Utilisation rapide

1. Choisissez le modèle (`tiny`, `base`, `small`, `medium`, `large-v3`).
2. Créez/sélectionnez un dossier de travail à gauche.
3. Glissez-déposez un fichier (`.mp3`, `.mp4`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.mov`, `.mkv`, etc.) dans la zone prévue.
4. Ou enregistrez depuis le micro (`Démarrer micro` puis `Arrêter micro`).
5. Cliquez `Transcrire`.
6. Le texte est sauvegardé dans le dossier sélectionné (`.txt` + `.json`).

## Benchmark local (optionnel)

```bash
python benchmark.py --file /chemin/vers/audio.mp3 --model small
```

Le script affiche:
- durée audio,
- temps de transcription,
- ratio temps réel (RTF = temps_transcription / durée_audio).

Plus le RTF est bas, plus c'est rapide.

## Structure de sortie

Par dossier de session :
- média source (si copié/importé)
- `transcription.txt`
- `transcription.json`

## Évolution SaaS (prochaine étape)

Cette base sépare déjà :
- couche GUI,
- couche transcription,
- gestion des sessions/fichiers.

Cela facilite la migration vers une API backend (FastAPI) + frontend web.
