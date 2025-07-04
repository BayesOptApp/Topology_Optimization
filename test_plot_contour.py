import os
from iohinspector import DataManager, plot_ecdf
import iohinspector
import polars as pl
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


from postprocessing.cantilever_cases import set_case_per_name, get_list_of_strings_of_variables


# Creating a data manager
manager:DataManager = DataManager()
data_folders = [os.path.join('Figures_Python', x) for x in os.listdir('Figures_Python')]


# Loop all over the data folders and exclude those without a JSON file
data_folders = [folder for folder in data_folders if any(file.endswith('.json') for file in os.listdir(folder))]
#manager.add_folders(data_folders[-1:])

manager.add_folders(data_folders[20:21])



print(manager.overview)

# Get the 
dt:pl.DataFrame = manager.select().load(False, True)

#some_name = dt['function_name'][0]
some_name = "Topology_Optimization_MMC"

# Get the design
design = set_case_per_name(name=some_name, material_definition="orthotropic")

# Get the list of variables
list_of_variables = get_list_of_strings_of_variables(dim =design.problem_dimension)

# Extract the data from the DataFrame
# Get the x values with best function value

#x_values = dt[list_of_variables][419].to_numpy()
x_values = np.asarray([0.13672694903794352, 0.2049985640530812, 0.06873562651187165, 0.8483222526442988, 0.3031933757682419, 0.8318830940761535, 0.998006665130156, 0.8794365297326651, 0.7650157988014814, 0.5345173139073329, 0.9302716621286777, 0.6881332079047313, 0.10113647882033772, 0.5952714233355805, 0.1017812403859123])

design.modify_mutable_properties_from_array(x_values, scaled=True, repair_level=0)


# Get the contour
fig, ax= design.plot_discrete_design()

plt.show()