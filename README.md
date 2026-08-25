# Projet Lung-MAP

Ce dépôt github contient le code source, les notebooks et les scripts associés aux prémices du projet Lung-MAP. Ce projet s'articule autour de deux axes : la caractérisation de la répartition tumorale et la segmentation d'images histologiques.

## Structure

L'arborescence du projet est organisée de la manière suivante :

```text
├── data/                   # Dossier destiné à accueillir les données
├── notebook/               # Notebooks d'exploration, de visualisation et d'analyse
├── qupath_scripts/         # Scripts utilisés pour l'extraction des annotations et des images histologiques via QuPath
└── src/                    # Code source Python du projet
    ├── repartition/        # Fichiers et modules relatifs à l'axe "Répartition"
    └── segmentation/       # Fichiers et modules relatifs à l'axe "Segmentation"
```

## Accès aux données et ressources

- **Données utilisés :** Les données de travail sont disponibles via ce [lien](https://drive.google.com/drive/folders/1ZWUNtICIBOt_bHIg_HwND1xcrKl8Idl-?hl=fr).
- **Données originales pour la segmentation :** [LUAD-HistoSeg](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr),  [BCSS](https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss).
- **Ressources produites :** Les documents, images, rapports et autres éléments produits durant ce projet sont accessible [ici](https://drive.google.com/drive/folders/1jqOBCDpHWad2nXvuYeeIDIv-ZfsCv9Wa?hl=fr).

**Instructions de mise en place des données :** 
Une fois les données téléchargées depuis google drive, veuillez les extraire et les placer dans le répertoire `data/` en respectant l'arborescence initiale pour que les scripts s'exécutent correctement.

## Installation

### Installation via `uv`

1. Clonez le dépôt :
  
  ```bash
  git clone https://github.com/JadeAffolabi/lung_map.git
  cd lung_map
  ```
  
2. Créez un environnement virtuel et installez les dépendances (via `pyproject.toml` ou `requirements.txt`) :
  
  ```bash
  uv venv
  source .venv/bin/activate

  uv sync
  
  # Ou si vous avez un fichier requirements.txt :
  # uv pip install -r requirements.txt
  ```
  

### Installation via `pip`

Si vous préférez utiliser l'outil standard `pip` :

1. Clonez le dépôt :
  
  ```bash
  git clone https://github.com/JadeAffolabi/lung_map.git
  cd lung_map
  ```
  
2. Créez et activez un environnement virtuel :
  
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
  
3. Installez les dépendances :

  ```bash
  pip install -r requirements.txt
  ```