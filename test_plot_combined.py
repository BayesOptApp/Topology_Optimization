#import tikzplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # Use pdflatex
    "text.usetex": True,          # Use LaTeX for text
    "font.family": "serif",       # Match LaTeX font
    "pgf.rcfonts": False,         # Do not override rc fonts
})

import numpy as np
import polars as pl

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

# Define colors for each algorithm
colors = {
    "CMA-ES": "blue",
    "turbo-1": "orange",
    "turbo-m": "green",
    "DE": "red",
    "Vanilla-BO": "purple",
    "BAxUS": "brown",
    "HEBO": "magenta"
}

exp_types = ("Concurrent", "Sequential")
material_definitions = ("isotropic", "orthotropic")
OFFICIAL_MATERIAL_DEFINITION_IDX = {"Concurrent": 0, "Sequential": 1}

fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
legend_lines = []
legend_labels = []

for i, material in enumerate(material_definitions):
    ax = axs[i]
    ax.set_title(f"{material.capitalize()} Material")
    
    for exp_type in exp_types:
        material_idx = OFFICIAL_MATERIAL_DEFINITION_IDX[exp_type]

        for alg in algorithms:
            base_path = ROOT_FOLDER / exp_type / material / alg
            folders = [base_path / x for x in os.listdir(base_path) if (base_path / x).is_dir()]
            folders = [f for f in folders if any(fname.endswith(".json") for fname in os.listdir(f))]

            if not folders:
                continue

            manager = DataManager()
            manager.add_folders(folders)
            manager_sub = manager.select(algorithms=[alg])

            if exp_type == "Sequential":
                manager_dim_15 = manager_sub.select(dimensions=[15])
                manager_dim_3 = manager_sub.select(dimensions=[3])

                dt_1 = manager_dim_15.select().load(False, True)
                dt_1 = turbo_align(dt_1, np.arange(1, 834), y_col="current_y", maximization=False)

                dt_2 = manager_dim_3.select().load(False, True)
                dt_2 = turbo_align(dt_2, np.arange(1, 166), y_col="current_y", maximization=False)
                dt_2 = dt_2.with_columns([
                    (pl.col("evaluations") + 833).alias("evaluations"),
                    (pl.col("data_id") - 1).alias("data_id")
                ])

                last_dt1 = dt_1.group_by("data_id").agg(
                    pl.col("current_y_best").last().alias("last_y_best")
                )
                dt_2 = dt_2.join(last_dt1, on="data_id", how="left")
                dt_2 = dt_2.with_columns(
                    pl.when(pl.col("current_y_best") > pl.col("last_y_best"))
                    .then(pl.col("last_y_best"))
                    .otherwise(pl.col("current_y_best"))
                    .alias("current_y_best")
                ).select(dt_2.columns)

                combined = pl.concat([dt_1, dt_2], how="diagonal")
            else:
                combined = manager_sub.select().load(False, True)
                combined = turbo_align(combined, np.arange(1, 1001), y_col="current_y", maximization=False)

            stats = combined.group_by("evaluations").agg([
                pl.mean("current_y_best").alias("mean"),
                pl.std("current_y_best").alias("std"),
                pl.len().alias("count")
            ]).sort("evaluations")

            evals = stats["evaluations"].to_numpy()
            mean = stats["mean"].to_numpy()
            std = stats["std"].to_numpy()
            count = stats["count"].to_numpy()
            ci = 1.96 * std / np.sqrt(count)

            label = f"{algorithm_labels[alg]} ({exp_type})"
            linestyle = "-" if exp_type == "Concurrent" else "--"

            line, = ax.plot(evals, mean, label=label, linestyle=linestyle, color=colors[alg])
            #ax.fill_between(evals, mean - ci, mean + ci, alpha=0.15, linestyle=linestyle)
            if label not in legend_labels:
                legend_lines.append(line)
                legend_labels.append(label)

    ax.axvline(x=833, color='gray', linestyle=':')
    ax.set_xlabel("Evaluations")
    ax.set_yscale('log')
    ax.set_xlim(0, 1000)

axs[0].set_ylabel("Compliance [best-so-far]")
fig.legend(legend_lines, legend_labels, loc='lower center', ncol=6, fontsize=9)
#plt.suptitle("Convergence for Isotropic vs Orthotropic Materials")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()


# Save the figure

#plt.savefig("convergence_plot_isotropic.pgf")
