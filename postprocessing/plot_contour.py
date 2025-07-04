import os
from iohinspector import DataManager, plot_ecdf
import iohinspector
import polars as pl
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from Design_Examples.Raw_Design.Design import Design

# Creating a data manager
manager = DataManager()
data_folders = [os.path.join('Figures_Python', x) for x in os.listdir('Figures_Python')]


# Loop all over the data folders and exclude those without a JSON file
data_folders = [folder for folder in data_folders if any(file.endswith('.json') for file in os.listdir(folder))]

manager.add_folders(data_folders[0])



print(manager.overview)