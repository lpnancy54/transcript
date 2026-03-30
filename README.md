# Transcripteur GUI (local, prêt pour SaaS)

Application Python avec interface graphique pour :

- enregistrer depuis le microphone du PC,
- **choisir le micro système** à utiliser,
- visualiser un **VU-mètre** en direct pendant l'enregistrement,
- importer des fichiers audio/vidéo par glisser-déposer,
- transcrire avec **faster-whisper**,
- valider la transcription,
- générer un **compte-rendu médical** éditable en **DOCX** ou exportable en **PDF**.

## Fonctionnalités médicales ajoutées

Après transcription :

1. Validation manuelle de la transcription (`Valider la transcription`).
2. Préremplissage automatique (si détecté) :
   - Nom patient
   - Prénom patient
   - Date de naissance
   - Motif de consultation
3. Édition manuelle des champs si non détectés.
4. Génération du rapport structuré :
   - Identification patient
   - Motif de consultation
   - Discussion patient/parents/praticien (si identifiable)
   - Symptômes et anamnèse
   - Explications et conduite à tenir
   - Autres praticiens mentionnés
5. Export en `.docx` ou `.pdf`.

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
./run_local.sh
```

Le script crée automatiquement `.venv`, installe les dépendances, puis lance l'application.

## Structure de sortie

Par dossier de session :
- média source (si copié/importé)
- `transcription.txt`
- `transcription.json`
- `Compte_rendu_<Nom>_<Prénom>.docx` ou `.pdf`

## Benchmark local (optionnel)

```bash
python benchmark.py --file /chemin/vers/audio.mp3 --model small
```

Le script affiche le ratio temps réel (RTF) pour valider la performance sur votre machine.
