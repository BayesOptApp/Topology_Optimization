import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# Import plotly for the parallel axis plot
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

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
MATERIAL_DEF = 1
OFFICIAL_MATERIAL_DEFINITION_IDX = {"Concurrent": 0, "Sequential": 1}
CONSTANTS = {
    "x15": 0.5,
    "x16": 0.0,
    "x17": 0.0
}

x_label_map = {
    'x0': '$x_{c,1}$',
    'x1': '$y_{c,1}$',
    'x2': '$\\theta_{1}$',
    'x3': '$l_1$',
    'x4': '$t_1$',
    'x5': '$x_{c,2}$',
    'x6': '$y_{c,2}$',
    'x7': '$\\theta_{2}$',
    'x8': '$l_2$',
    'x9': '$t_2$',
    'x10': '$x_{c,3}$',
    'x11': '$y_{c,3}$',
    'x12': '$\\theta_{3}$',
    'x13': '$l_3$',
    'x14': '$t_3$',
    'x15': '$V_{r}$',
    'x16': '$V_{3,1}$',
    'x17': '$V_{3,2}$'
}


global_results = []

for exp_type in exp_types:
    material_idx = OFFICIAL_MATERIAL_DEFINITION_IDX[exp_type]
    results = []
    for alg in algorithms:
        # Build folder paths
        base_path = ROOT_FOLDER / exp_type / material_definitions[MATERIAL_DEF] / alg
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
algorithm_order = [algorithm_labels[alg] for alg in algorithms]

# Create a mapping from algorithm name to color
color_discrete_map = {
    algorithm_labels[alg]: color
    for alg, color in zip(algorithms, ['blue','purple', 'orange', 'cyan', 'red', 'green', 'magenta'])
}


mixed_data_1['algorithm_num'] = mixed_data_1['algorithm_name'].map(class_map)
mixed_data_2['algorithm_num'] = mixed_data_2['algorithm_name'].map(class_map)

# Map the algorithm names to the official labels
mixed_data_1['algorithm_name'] = mixed_data_1['algorithm_name'].map(algorithm_labels)
mixed_data_2['algorithm_name'] = mixed_data_2['algorithm_name'].map(algorithm_labels)

# Add experiment type as a column
mixed_data_1["exp_type"] = "Concurrent"
mixed_data_2["exp_type"] = "Sequential"

# Combine both into one DataFrame
combined_df = pd.concat([mixed_data_1, mixed_data_2], ignore_index=True)

# If current_y_best isn't in final df, fallback to current_y
if "current_y_best" not in combined_df.columns and "current_y" in combined_df.columns:
    combined_df["current_y_best"] = combined_df["current_y"]

# Compute global best value
best_global_value = combined_df["current_y_best"].min()

# Add a column for the performance gap
combined_df["gap_to_best"] = combined_df["current_y_best"] - best_global_value

# Melt to long format (if needed, not necessary here since we only plot one column)
df_gap = combined_df[["algorithm_name", "exp_type", "current_y_best", "gap_to_best"]].copy()

# Sort by algorithm name for better visualization
df_gap["algorithm_name"] = pd.Categorical(
    df_gap["algorithm_name"],
    categories=algorithm_order,
    ordered=True
)

# # Plot: Box plot of gaps per algorithm, grouped by mode
# fig = px.box(
#     df_gap,
#     x="algorithm_name",
#     y="current_y_best",
#     color="exp_type",
#     title="Modified Compliance Value Distribution by Algorithm and Strategy after full budget",
#     labels={
#         "current_y_best": "Compliance",
#         "algorithm_name": "Algorithm",
#         "exp_type": "Mode"
#     },
#     range_y=(4e-02,70),
#     log_y=True,
#     points=False,
    
# )

# # Step 1: Compute mean compliance per (algorithm_name, exp_type)
# means = (
#     df_gap.groupby(["algorithm_name", "exp_type"])["current_y_best"]
#     .mean()
#     .reset_index()
# )

# # Step 2: Build compound x labels used by plotly express internally
# means["x_label"] = means["algorithm_name"].astype(str)




# fig.update_layout(
#     boxmode="group",
#     xaxis_title="Algorithm",
#     yaxis_title="Modified Compliance",
#     legend_title="Optimization Approach"
# )

# # Step 3: Add dashed horizontal lines to indicate the mean for each box
# # for _, row in means.iterrows():
# #     fig.add_shape(
# #         type="line",
# #         x0=row["x_label"],
# #         x1=row["x_label"],
# #         y0=row["current_y_best"],
# #         y1=row["current_y_best"],
# #         xref="x",
# #         yref="y",
# #         line=dict(color="red", width=2, dash="dash")
# #     )

# fig.add_trace(go.Scatter(
#     x=means["x_label"],
#     y=means["current_y_best"],
#     mode="markers",
#     marker=dict(color="orange", size=8, symbol="x"),
#     name="Mean",
#     showlegend=True
# ))

# Create an empty figure
fig = go.Figure()

# Define your modes and a color map
modes = ["Concurrent", "Sequential"]
colors = {
    "Concurrent": "blue",
    "Sequential": "red"
}

# Plot each (algorithm, mode) pair with appropriate offsets
for mode in modes:
    df_mode = df_gap[df_gap["exp_type"] == mode]
    for algorithm in df_gap["algorithm_name"].cat.categories:
        y_vals = df_mode[df_mode["algorithm_name"] == algorithm]["current_y_best"]
        if not y_vals.empty:
            fig.add_trace(go.Box(
                y=y_vals,
                x=[algorithm] * len(y_vals),  # group by algorithm
                name=mode,  # will appear in legend
                marker_color=colors[mode],
                boxpoints=False,
                legendgroup=mode,
                offsetgroup=mode,
                boxmean=True,  # show mean
                showlegend=(algorithm == df_gap["algorithm_name"].cat.categories[0])  # one legend per mode
            ))

# Layout
fig.update_layout(
    #title="Modified Compliance Value Distribution by Algorithm and Strategy after full budget",
    yaxis_title="Modified Compliance",
    xaxis_title="Algorithm",
    legend_title="Optimization Approach",
    boxmode="group",  # Important: enables grouping by offsetgroup
    yaxis_type="log",
    yaxis_range=[np.log10(0.04), np.log10(70)],  # equivalent to range_y=(4e-02, 70)
)

fig.show()
#fig.write_image("arfito2.pdf")