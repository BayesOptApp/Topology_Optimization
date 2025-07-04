import os
from iohinspector import DataManager, plot_ecdf
import iohinspector
import polars as pl
import pandas as pd

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.cantilever_cases import set_case_per_name, get_list_of_strings_of_variables

# ROOT FOLDER
ROOT_FOLDER = Path("C:/Users/iolar/Downloads/Final_Repo_Paper")

# OFFICIAL MATERIAL DEFINITION
OFFICIAL_MATERIAL_DEFINITION_IDX = 0  # 0 for isotropic, 1 for orthotropic

# Experiment Types
exp_types = ("No-LP","Concurrent","Sequential")
material_definitions = ("isotropic", "orthotropic")
algorithms = ("CMA-ES","TuRBO-1","TuRBO-m","BAxUS","HEBO","Vanilla-BO","DE")

# Creating a data manager
manager:DataManager = DataManager()
# Get all subdirectories in the base path

base_paths = [ROOT_FOLDER.joinpath(exp_types[1], material_definitions[OFFICIAL_MATERIAL_DEFINITION_IDX], algorithms[jj]) for jj in range(len(algorithms))]

data_folders = [
    base_path / x
    for base_path in base_paths
    for x in os.listdir(base_path)
    if (base_path / x).is_dir()
]

# Keep only folders that contain at least one JSON file
data_folders = [
    folder for folder in data_folders
    if any(file.endswith('.json') for file in os.listdir(folder))
]
#manager.add_folders(data_folders[-1:])

manager.add_folders(data_folders)


print(manager.overview)

fig, ax = plt.subplots(figsize=(10, 6))

for alg in ["CMA-ES","turbo-1","turbo-m", "DE", "Vanilla-BO", "BAxUS", "HEBO"]:
    manager_sub = manager.select(algorithms=[alg])

    dt = manager_sub.select().load(False, True)

    #dt = dt.with_columns(pl.col("current_y_best").cum_max().over("run_id"))


    dt = iohinspector.turbo_align(dt, np.arange(1, 1001, 1), y_col='current_y', maximization=False)   


    df_pd:pd.DataFrame = dt.to_pandas()


    

    # Filter per algorithm
    df_2_alg = df_pd[df_pd["algorithm_name"] == alg]

    # Ensure numeric type
    df_2_alg["current_y_best"] = df_2_alg["current_y_best"].astype(float)

    # Group by evaluations and compute mean and std
    grouped = df_2_alg.groupby("evaluations")["current_y_best"]
    mean_df_alg = grouped.mean().reset_index()
    std_df_alg = grouped.std().reset_index()
    count = grouped.count().reset_index()

    ax.plot(
        mean_df_alg["evaluations"],
        mean_df_alg["current_y_best"],
        label=alg
    )
    ax.fill_between(
        mean_df_alg["evaluations"],
        mean_df_alg["current_y_best"] - 1.96*std_df_alg["current_y_best"]/np.sqrt(20),
        mean_df_alg["current_y_best"] + 1.96*std_df_alg["current_y_best"]/np.sqrt(20),
        alpha=0.2
    )

ax.set_xlabel("Evaluations")
ax.set_ylabel("best-so-far")
ax.set_xlim(80, 1000)
#.set_ylim(-.00005, 10**4)
#ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()
ax.set_title("Convergence Plots for Concurrent Optimisation (Isotropic Material)")
plt.tight_layout()
plt.show()

# Get the 
#dt:pl.DataFrame = manager.select().load(False, True)

#some_name = dt['function_name'][0]
#some_name = "Topology_Optimization_MMC"

# Get the design
#design = set_case_per_name(name=some_name, material_definition=material_definitions[OFFICIAL_MATERIAL_DEFINITION_IDX])

# Get the list of variables
#list_of_variables = get_list_of_strings_of_variables(dim =design.problem_dimension)

# Extract the data from the DataFrame
# Get the x values with best function value

#x_values = dt[list_of_variables][3105].to_numpy()
#x_values = np.asarray([0.13672694903794352, 0.2049985640530812, 0.06873562651187165, 0.8483222526442988, 0.3031933757682419, 0.8318830940761535, 0.998006665130156, 0.8794365297326651, 0.7650157988014814, 0.5345173139073329, 0.9302716621286777, 0.6881332079047313, 0.10113647882033772, 0.5952714233355805, 0.1017812403859123])

#design.modify_mutable_properties_from_array(x_values, scaled=True, repair_level=0)


# Get the contour
#fig, ax= design.plot_discrete_design()

#plt.show()