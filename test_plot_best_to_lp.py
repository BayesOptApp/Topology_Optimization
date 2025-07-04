import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd

# Import plotly for the parallel axis plot
from postprocessing.cantilever_cases import set_case_per_name
#pio.renderers['browser'].browser = 'edge'
#pio.renderers.default = 'vscode'

from iohinspector import DataManager, turbo_align
from pathlib import Path
import os

ROOT_FOLDER = Path("C:/Users/iolar/Downloads/Final_Repo_Paper")
algorithms = ["CMA-ES","turbo-1","turbo-m", "DE", "Vanilla-BO", "BAxUS", "HEBO"]
algorithm_labels = {
    "CMA-ES": "CMA-ES",
    "turbo-1": "TuRBO-1",
    "turbo-m": "TuRBO-m",
    "DE": "DE",
    "Vanilla-BO": "Vanilla-BO",
    "BAxUS": "BAxUS",
    "HEBO": "HEBO"
}

exp_types = ("Concurrent", "Sequential")
material_definitions = ("isotropic", "orthotropic")
MAT_DEF = 1
OFFICIAL_MATERIAL_DEFINITION_IDX = {"Concurrent": 0, "Sequential": 1}
CONSTANTS = {
    "x15": 0.5,
    "x16": 0.0,
    "x17": 0.0
}

x_label_map = {
    'x0': 'x꜀₁',
    'x1': 'y꜀₁',
    'x2': 'θ₁',
    'x3': 'l₁',
    'x4': 't₁',
    'x5': 'x꜀₂',
    'x6': 'y꜀₂',
    'x7': 'θ₂',
    'x8': 'l₂',
    'x9': 't₂',
    'x10': 'x꜀₃',
    'x11': 'y꜀₃',
    'x12': 'θ₃',
    'x13': 'l₃',
    'x14': 't₃',
    'x15': 'Vᵣ',
    'x16': 'V₃₁',
    'x17': 'V₃₂',
}

def extract_grouped_best_and_median(data: pd.DataFrame, exp_type_label: str) -> pd.DataFrame:
    rows = []
    for alg in sorted(data['algorithm_name'].unique()):
        subset = data[data['algorithm_name'] == alg]
        if subset.empty:
            continue

        # Sort by current_y
        sorted_subset = subset.sort_values("current_y").reset_index(drop=True)

        # Best result (first)
        best_row = sorted_subset.iloc[0].copy()
        best_row["performance_type"] = "best"
        best_row["exp_type"] = exp_type_label

        # Median result (middle)
        median_idx = len(sorted_subset) // 2
        median_row = sorted_subset.iloc[median_idx].copy()
        median_row["performance_type"] = "median"
        median_row["exp_type"] = exp_type_label

        rows.extend([best_row, median_row])

    return pd.DataFrame(rows)

global_results = []

for exp_type in exp_types:
    material_idx = OFFICIAL_MATERIAL_DEFINITION_IDX[exp_type]
    results = []
    for alg in algorithms:
        # Build folder paths
        base_path = ROOT_FOLDER / exp_type / material_definitions[MAT_DEF] / alg
        folders = [base_path / x for x in os.listdir(base_path) if (base_path / x).is_dir()]
        folders = [f for f in folders if any(fname.endswith(".json") for fname in os.listdir(f))]

        if not folders:
            continue

        # Load with IOHInspector
        manager = DataManager()
        manager.add_folders(folders)
        manager_sub = manager.select(algorithms=[alg])
        
        # Sequential needs merge logic
        if exp_type == "Sequential":
            manager_dim_15 = manager_sub.select(dimensions=[15])
            manager_dim_3 = manager_sub.select(dimensions=[3])

            dt_1 = manager_dim_15.select().load(False, True)
            dt_1 = turbo_align(dt_1, np.arange(1, 834, 1), y_col="current_y", maximization=False)

            dt_1 = dt_1.with_columns([
              pl.lit(v).alias(k) for k, v in CONSTANTS.items()
            ])

            

            dt_2 = manager_dim_3.select().load(False, True)
            dt_2 = turbo_align(dt_2, np.arange(1, 168, 1), y_col="current_y", maximization=False)
            dt_2 = dt_2.with_columns([
                (pl.col("evaluations") + 833).alias("evaluations"),
                (pl.col("data_id") - 1).alias("data_id")
            ])

            # Rename the columns for clarity
            dt_2 = dt_2.rename({"x0": "x15",
                                "x1": "x16", "x2": "x17"})

            # Merge best values from dt_1 into dt_2
            last_dt1 = dt_1.group_by("data_id").agg(
                pl.col("current_y_best").last().alias("last_y_best")
            )

            # Step 1: Get the best (minimum current_y) row per data_id with corresponding x0–x14
            best_xs = (
                dt_1.sort("current_y")  # ensure lowest y is on top
                    .group_by("data_id")
                    .agg([
                        *[pl.col(f"x{i}").first().alias(f"x{i}") for i in range(15)]
                    ])
            )

            # Step 2: Join into dt_2 and broadcast these best values across all rows per data_id
            #dt_2 = dt_2.drop([f"x{i}" for i in range(15)])  # remove old x0–x14 if they exist

            dt_2 = dt_2.join(best_xs, on="data_id", how="left")
            dt_2 = dt_2.join(last_dt1, on="data_id", how="left")
            dt_2 = dt_2.with_columns(
                pl.when(pl.col("current_y_best") > pl.col("last_y_best"))
                  .then(pl.col("last_y_best"))
                  .otherwise(pl.col("current_y_best"))
                  .alias("current_y_best")
            ).select(dt_2.columns)

            # Delete column last_y_best
            dt_2 = dt_2.drop("last_y_best")

            combined = pl.concat([dt_1, dt_2], how="align")
        else:
            combined = manager_sub.select().load(False, True)
            combined = turbo_align(combined, np.arange(1, 1001), y_col="current_y", maximization=False)

        # combined = combined.filter(pl.col("algorithm_name") == alg)

        # Add row index
        combined = combined.with_row_index(name="row_index")

        # Get min current_y_best per data_id
        min_per_group = combined.group_by("data_id").agg(
            pl.col("current_y").min().alias("min_y")
        )

        # Join to original to get full rows where the min occurs
        #best_rows = combined.join(min_per_group, on="data_id")
        best_rows = (
            combined
            .join(min_per_group, on="data_id")
            .filter(pl.col("current_y") == pl.col("min_y"))
            .sort("evaluations", descending=True)
            .unique(subset=["data_id"])
        )

        # Keep relevant columns: algorithm, exp_type, row index, etc.
        

        results.append(best_rows)

        # results.append({
        #     "exp_type": exp_type,
        #     "material": mat_def,
        #     "algorithm": alg,
        #     "best_current_y": best_y
        # })
    
    global_results.append(results)



mixed_data_1 = global_results[0]
mixed_data_2 = global_results[1]

#Concatenate all results for the first experiment type
mixed_data_1:pl.DataFrame = pl.concat(mixed_data_1, how="align")
mixed_data_2:pl.DataFrame = pl.concat(mixed_data_2, how="align")

# Convert to pandas DataFrame for Plotly
mixed_data_1 = mixed_data_1.to_pandas()
mixed_data_2 = mixed_data_2.to_pandas()


# Step 1: Map classes to integers
class_map = {label: idx for idx, label in enumerate(algorithms)}
algorithm_order = list(class_map.items())
mixed_data_1['algorithm_num'] = mixed_data_1['algorithm_name'].map(class_map)
mixed_data_2['algorithm_num'] = mixed_data_2['algorithm_name'].map(class_map)

# Map the algorithm names to the official labels
mixed_data_1['algorithm_name'] = mixed_data_1['algorithm_name'].map(algorithm_labels)
mixed_data_2['algorithm_name'] = mixed_data_2['algorithm_name'].map(algorithm_labels)


# Extract from both datasets
summary_df_1 = extract_grouped_best_and_median(mixed_data_1, "Concurrent")

# Save the summary DataFrame to a CSV file
summary_df_1.to_csv(f"summary_concurrent_{material_definitions[MAT_DEF]}.csv", index=False)

summary_df_2 = extract_grouped_best_and_median(mixed_data_2, "Sequential")
summary_df_2.to_csv(f"summary_sequential_{material_definitions[MAT_DEF]}.csv", index=False)

#print("Summary DataFrame for Concurrent:")
#print(summary_df_1.head())

# Get the design object
design_lp = set_case_per_name("Topology_Optimization_With_Lamination_Parameters", material_definition=[MAT_DEF])

# Loop all over summary DataFrames per row
for idx, row in summary_df_1.iterrows():
    # Get the algorithm name and performance type
    alg_name = row['algorithm_name']
    perf_type = row['performance_type']
    
    
    # Extract the x-values from the row
    x_values = row[[f'x{i}' for i in range(18)]].values

    # Change the internal values to the design object
    x_values = design_lp.modify_mutable_properties_from_array(x_values, scaled=True,
                                                              repair_level=0)
    
    # Plot the design
    design_lp.evaluate_FEA_design(volfrac=0.5,
                                  iterr=row['evaluations'],
                                  sample=0,
                                  run_=row['data_id'],
                                  use_sparse_matrices=True,
                                  plotVariables=True,
                                  cost_function="compliance",
                                  avoid_computation_for_not_compliance=False)

# Loop all over summary DataFrames per row
for idx, row in summary_df_2.iterrows():
    # Get the algorithm name and performance type
    alg_name = row['algorithm_name']
    perf_type = row['performance_type']
    
    
    # Extract the x-values from the row
    x_values = row[[f'x{i}' for i in range(18)]].values

    # Change the internal values to the design object
    x_values = design_lp.modify_mutable_properties_from_array(x_values, scaled=True,
                                                              repair_level=0)
    
    # Plot the design
    design_lp.evaluate_FEA_design(volfrac=0.5,
                                  iterr=row['evaluations'],
                                  sample=0,
                                  run_=row['data_id']+2000,
                                  use_sparse_matrices=True,
                                  plotVariables=True,
                                  cost_function="compliance",
                                  avoid_computation_for_not_compliance=False)


