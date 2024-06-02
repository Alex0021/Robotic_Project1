import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

DATA_FILE_DIR = "../hardware_results"
EXPORT_FOLDER = "../exported_plots"
DATA_FILENNAME = "data.csv"


if __name__ == "__main__":
    # Read arguments
    nb_args = len(sys.argv)
    if nb_args < 2:
        print('Please provide at least one argument --> --visual or --export')
        sys.exit(1)
    
    cmd = sys.argv[1]
    folder = DATA_FILE_DIR
    export_mode = False
    if nb_args == 3:
        folder = os.path.join(DATA_FILE_DIR, sys.argv[2])
    if cmd == '--visual':
        print(f'Visualizing plots of folder "{folder}"')
    elif cmd == '--export':
        export_mode = True
        os.makedirs(EXPORT_FOLDER, exist_ok=True)
        print(f'Exporting plots of folder "{folder}" to "{EXPORT_FOLDER}"')
    else:
        print('Please provide a valid argument --> --visual or --export')
        sys.exit(1)
    
    # Load the dataset
    data = pd.read_csv(os.path.join(folder, DATA_FILENNAME))

    print('Data columns: ', data[data.__time <= 3])