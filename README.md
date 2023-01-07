# painter_image_prediction

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=545509676&machine=basicLinux32gb&location=WestEurope)


## Documentation

The summary of what models/improvements are used are in [docs](docs/README.md)

## Setup for codespaces

1. Open an new codespace with the link above
2. Make an new file `.env` in the root directory (the contents from this file have been send via toledo)
3. Open the terminal (top left -> view -> terminal)
4. `pip install -e requirements`
5. `python3 website.py`
6. Open the website (popup will show on the bottom right)

### Debugging
- if the website.py script doesn't show you can open the ports
  - Press ctrl + p and type `> ports`
  - Select option `view toggle ports`
  - Open port 8080
## Setup for training 

1. Copy data folders to data/raw
2. `conda env create -f environment.yml`
3. `conda activate painters`
4. `python3 preprocessing.py`
5. `cd docker;docker compose up -d --build`
6. `python3 train.py`
