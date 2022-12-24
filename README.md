# painter_image_prediction
## Documentation

The summary of what models is used is in [docs](docs/README.md)
## Setup 

1. Copy data folders to data/raw
2. `conda env create -f environment.yml`
3. `conda activate painters`
4. `python3 preprocessing.py`
5. `cd docker;docker compose up -d --build`
6. `python3 train.py`