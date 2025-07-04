import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Import plotly for the parallel axis plot
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

from iohinspector import DataManager, turbo_align
from pathlib import Path
import os

ROOT_FOLDER = Path("C:/Users/iolar/Downloads/Final_Repo_Paper")
algorithms = ["Vanilla-BO","CMA-ES","turbo-1","turbo-m", "DE", "BAxUS", "HEBO"]
algorithm_labels = {
    "CMA-ES": "CMA-ES",
    "turbo-1": "TuRBO-1",
    "turbo-m": "TuRBO-m",
    "DE": "DE",
    "Vanilla-BO": "Vanilla-BO",
    "BAxUS": "BAxUS",
    "HEBO": "HEBO"
}

exp_types = ("Concurrent","Sequential")
material_definitions = ("isotropic", "orthotropic")
OFFICIAL_MATERIAL_DEFINITION_IDX = {"Concurrent": 0, "Sequential": 1}
CONSTANTS = {
    "x15": 0.5,
    "x16": 0.0,
    "x17": 0.0
}


global_results = []

for exp_type in exp_types:
    material_idx = OFFICIAL_MATERIAL_DEFINITION_IDX[exp_type]
    results = []
    for alg in algorithms:
        # Build folder paths
        base_path = ROOT_FOLDER / exp_type / "isotropic" / alg
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

the_dims = []
for ii in range(15):
    col = f'x{ii}'
    if col in mixed_data_1.columns:
        the_dims.append(dict(range=[0, 1], label=col, values=mixed_data_1[col]))
    else:
        print(f"Column {col} is missing in the DataFrame.")

# Generate the parallel axis plot
fig1 = go.Figure(data=
    go.Parcoords(
        line = dict(color = mixed_data_1['current_y_best'],
                   colorscale = 'Electric',
                   cmin = 4e-02,
                   cmax = 0.2,
                   showscale = True,
                   reversescale=True),
        dimensions = the_dims
)
)
fig1.show()

# Generate the parallel axis plot
fig2 = go.Figure(data=
    go.Parcoords(
        line = dict(color = mixed_data_2['current_y_best'],
                   colorscale = 'Electric',
                   cmin = 4e-02,
                   cmax = 0.2,
                   showscale = True,
                   reversescale=True),
        dimensions = the_dims
)
)
fig2.show()
        

# # Formatting
# ax.axvline(x=833, color='gray', linestyle=':')
# ax.set_xlabel("Evaluations")
# ax.set_ylabel("best-so-far")
# ax.set_yscale('log')
# ax.set_xlim(80, 1000)
# ax.legend()
# ax.set_title("Convergence of Sequential vs Concurrent Optimization")
# plt.tight_layout()
# plt.show()