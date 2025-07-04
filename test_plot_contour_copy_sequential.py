import os
from iohinspector import DataManager, plot_ecdf
import iohinspector
import polars as pl
import pandas as pd

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.cantilever_cases import set_case_per_name, get_list_of_strings_of_variables


def replace_if_worse_by_data_id(dt_1: pl.DataFrame, dt_2: pl.DataFrame) -> pl.DataFrame:
    """
    For each data_id, replace 'current_y_best' in dt_2 with the last value from dt_1 
    if the dt_2 value is worse (i.e., greater than dt_1's last value).

    Parameters:
    - dt_1: Polars DataFrame with columns 'data_id' and 'current_y_best'.
    - dt_2: Polars DataFrame with columns 'data_id' and 'current_y_best'.

    Returns:
    - Updated dt_2 DataFrame with 'current_y_best' replaced where applicable.
    """

    # Step 1: Get last current_y_best per data_id from dt_1
    last_dt1 = dt_1.group_by("data_id").agg(
        pl.col("current_y_best").last().alias("last_y_best")
    )

    # Step 2: Join dt_2 with the last values from dt_1
    joined = dt_2.join(last_dt1, on="data_id", how="left")

    # Step 3: Replace current_y_best in dt_2 where it is worse than dt_1's last_y_best
    updated = joined.with_columns(
        pl.when(pl.col("current_y_best") > pl.col("last_y_best"))
          .then(pl.col("last_y_best"))
          .otherwise(pl.col("current_y_best"))
          .alias("current_y_best")
    )

    # Step 4: Return the updated DataFrame with original columns
    return updated.select(dt_2.columns)


# ROOT FOLDER
ROOT_FOLDER = Path("C:/Users/iolar/Downloads/Final_Repo_Paper")

# OFFICIAL MATERIAL DEFINITION
OFFICIAL_MATERIAL_DEFINITION_IDX = 1  # 0 for isotropic, 1 for orthotropic

# Experiment Types
exp_types = ("No-LP","Concurrent","Sequential")
material_definitions = ("isotropic", "orthotropic")
algorithms = ("CMA-ES","TuRBO-1","TuRBO-m","BAxUS","HEBO","Vanilla-BO","DE")

# Creating a data manager
manager:DataManager = DataManager()
# Get all subdirectories in the base path

base_paths = [ROOT_FOLDER.joinpath(exp_types[2], material_definitions[OFFICIAL_MATERIAL_DEFINITION_IDX], algorithms[jj]) for jj in range(len(algorithms))]

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


manager.add_folders(data_folders)


print(manager.overview)

fig, ax = plt.subplots(figsize=(10, 6))

for alg in ["CMA-ES","turbo-1","turbo-m", "DE", "Vanilla-BO", "BAxUS", "HEBO"]:
    
    manager_sub = manager.select(algorithms=[alg])

    manager_dim_15 = manager_sub.select(dimensions=[15])
    manager_dim_3 = manager_sub.select(dimensions=[3])

    # Load the data
    dt_1:pl.DataFrame = manager_dim_15.select().load(False, True)
    dt_1 = iohinspector.turbo_align(dt_1, 
                                    np.arange(1, 834, 1), 
                                    y_col='current_y_best', 
                                    maximization=False) 
    dt_2:pl.DataFrame = manager_dim_3.select().load(False, True)
    dt_2 = iohinspector.turbo_align(dt_2, 
                                    np.arange(1, 168, 1), 
                                    y_col='current_y_best', 
                                    maximization=False) 

    # Add the column evaluations dt_2
    dt_2 = dt_2.with_columns((pl.col("evaluations") + 833).alias("evaluations"))

    # Subtract data id 
    dt_2 = dt_2.with_columns((pl.col("data_id") - 1).alias("data_id"))

    dt_2_bis = replace_if_worse_by_data_id(dt_1, dt_2)

    # Concatenate the two DataFrames
    combined_df = pl.concat([dt_1, dt_2_bis], 
                            how="diagonal")

    df_filtered = combined_df.filter(pl.col("algorithm_name") == alg)

    # Ensure 'current_y_best' is cast to float (if needed)
    # df_filtered = df_filtered.with_columns(
    #     pl.col("current_y_best").cast(pl.Float64)
    # )

    # Group and aggregate
    grouped_stats = df_filtered.group_by("evaluations").agg(
    pl.mean("current_y_best").alias("mean_y_best"),
    pl.std("current_y_best").alias("std_y_best"),
    pl.count().alias("count")
)

    # Sort by evaluations for plotting
    grouped_stats = grouped_stats.sort("evaluations")

    # Extract columns to numpy
    evals = grouped_stats["evaluations"].to_numpy()
    means = grouped_stats["mean_y_best"].to_numpy()
    stds = grouped_stats["std_y_best"].to_numpy()
    counts = grouped_stats["count"].to_numpy()

    # Compute confidence intervals
    conf_interval = 1.96 * stds / np.sqrt(counts)

    # Plot
    ax.plot(evals, means, label=alg)
    ax.fill_between(evals, means - conf_interval, means + conf_interval, alpha=0.2)
    #dt = dt.with_columns(pl.col("current_y_best").cum_max().over("run_id"))


    # dt = iohinspector.turbo_align(dt, np.arange(1, 1001, 1), y_col='current_y', maximization=False)   


    # df_pd:pd.DataFrame = dt.to_pandas()


    

    # # Filter per algorithm
    # df_2_alg = df_pd[df_pd["algorithm_name"] == alg]

    # # Ensure numeric type
    # df_2_alg["current_y_best"] = df_2_alg["current_y_best"].astype(float)

    # # Group by evaluations and compute mean and std
    # grouped = df_2_alg.groupby("evaluations")["current_y_best"]
    # mean_df_alg = grouped.mean().reset_index()
    # std_df_alg = grouped.std().reset_index()
    # count = grouped.count().reset_index()

    # ax.plot(
    #     mean_df_alg["evaluations"],
    #     mean_df_alg["current_y_best"],
    #     label=alg
    # )
    # ax.fill_between(
    #     mean_df_alg["evaluations"],
    #     mean_df_alg["current_y_best"] - 1.96*std_df_alg["current_y_best"]/np.sqrt(20),
    #     mean_df_alg["current_y_best"] + 1.96*std_df_alg["current_y_best"]/np.sqrt(20),
    #     alpha=0.2
    # )

# Plot vertical line at 833 evaluations
ax.axvline(x=833, color='gray', linestyle='--')

ax.set_xlabel("Evaluations")
ax.set_ylabel("best-so-far")
ax.set_xlim(80, 1000)
#ax.set_ylim(-.00005, 10**4)
#ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()
ax.set_title("Convergence Plots for Sequential Optimisation (Orthotropic Material)")
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