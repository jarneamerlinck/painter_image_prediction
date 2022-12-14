# Load packages
## Data wrangling
import pandas as pd
import numpy as np

## Helpers
from helpers.preprocessing_helpers import *

# Main preprocessing function
def main():
    # Save results to data/preprocessed
    painters = ["Picasso", "Rubens","Mondriaan", "Rembrandt"]
    make_data_sets(painters, 600, shape=(180, 180))
if __name__ == '__main__':
    main()
