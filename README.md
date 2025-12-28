# SNN-ACO

Hybrid Spiking Neural Network + Ant Colony Optimization experiments for bird vs drone classification.

## Setup
- Python 3.12 recommended.
- Create a virtual environment:
  - Windows: `python -m venv .venv && .venv\Scripts\activate`
  - Unix: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (or `pip install torch numpy pillow matplotlib` if you prefer manual install). Kaggle CLI is optional if you need to download data.

## Dataset
- Expected layout: `data/bird_drone/{train,val,test}/{bird,drone}/*.jpg`.
- If the dataset is included in your submission, ensure the above tree exists under `data/`.
- If not included (too large), provide a public link and instructions. Example using Kaggle CLI:
  1. Place your Kaggle API token at `.kaggle/kaggle.json` (do not commit this file).
  2. Install CLI: `pip install kaggle`.
  3. Download and unzip into `data/`:
     - `kaggle datasets download -d <dataset-slug> -p data --unzip`
     - If the archive unpacks to a different structure, move/rename into `data/bird_drone/train`, `val`, `test` with `bird` and `drone` subfolders.

## Run
- Baseline/main entry: `python main.py`
- Training script: `python train.py`
- Visualization utilities: `python visualize.py`

## Notes
- Do not commit `data/` or `.kaggle/`; they remain local only.
