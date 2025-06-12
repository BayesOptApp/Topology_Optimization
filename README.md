# Topology Optimization
## Introduction
This repository provides an implementation of **structural topology optimization** using the **Moving Morphable Components (MMC)** approach described in:

> Guo, X., Zhang, W., & Zhong, W. (2016). *Explicit structural topology optimization based on moving morphable components (MMC) with curved skeletons*. Computer Methods in Applied Mechanics and Engineering.

---

As of now, the current running code is based on a Concurrent Topology Optimization, wherein just not the topology is optimized, but lamination parameters with Fiber-Steering are optimized given the topology. Furthermore, the only constraints studied are the ones in which the structure is clamped on the left side and there is a point load to the middle-right. Currently, the topology is optimized by using Moving Morphable Components (MMC), an idea from Guo et al. [1], to reduce the dimensionality from the SIMP approach from Bendsoe & Sigmund [4]. For more information on the problem formulation and models behind, the following references are recommended:

[1] X. Guo, W. Zhang, J. Zhang, and J. Yuan, “Explicit structural topology optimization based on moving morphable components (MMC) with curved skeletons,” Computer Methods in Applied Mechanics and Engineering, vol. 310, pp. 711–748, Oct. 2016, doi: 10.1016/j.cma.2016.07.018.

[2] G. Serhat, “Concurrent Lamination and Tapering Optimization of Cantilever Composite Plates under Shear,” Materials, vol. 14, no. 9, p. 2285, Apr. 2021, doi: 10.3390/ma14092285.

[3] E. Raponi, M. Bujny, M. Olhofer, N. Aulig, S. Boria, and F. Duddeck, “Kriging-assisted topology optimization of crash structures,” Computer Methods in Applied Mechanics and Engineering, vol. 348, pp. 730–752, May 2019, doi: 10.1016/j.cma.2019.02.002.

[4] M. P. Bendsøe and O. Sigmund, Topology Optimization. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004. doi: 10.1007/978-3-662-05086-6.

The working code outputs two different plots, which are the deformed structure with the contours of the Von Mises Stress of the material and the distribution of lamination parameters (V1 and V3). A sample of these plots are shown below.
![image](https://github.com/user-attachments/assets/9105270f-19b4-4fd0-9579-9f0ff94f15e3)
![image](https://github.com/user-attachments/assets/02c6ea6e-1cc2-4806-b623-f6f76f9fb223)

## Required Packages

As of now, to run the code, you have to generate a new python environment. Via `pip`, you can install all the required dependencies via:

```markdown
python -m pip install -r requirements.txt
```

## 📂 Repository Structure

```text
.
├── main_example.py                  # Main script to reproduce results
├── problems/                        # Contains problem definitions (e.g., Cantilever) for the 2D plane-stress definitions found in (Guo et al., 2016)
│   ├── __init__.py                  
├── Algorithms/                      # Optimization algorithms
│   ├── __init__.py
│   └── cma_es_wrapper.py            # Wrapper for CMA-ES optimizer
├── Design_Examples/
│   ├── IOH_Wrappers/
│   │    ├──IOH_Wrapper_LP.py        # Holds the class which wraps the Design Problem with Lamination Parameters into IOH callable problems
│   │    └──IOH_Wrapper.py           # Holds the class which wraps the standard Design Problem into IOH callable problems.
│   ├── Raw_Design/
│   │   ├──Design_LP.py              # Holds the class which defines the Design Problem with Lamination Parameters.
│   │   └──Design.py                 # Holds the class which defines  standard Design Problem with MMC.
├── Figures_Python/                  # Output folder for IOH logs and figures
│   ├── Run_29/
│   │   ├── data_f0_{problem_name}   # The folder with the experimental results of a run. This follows the IOH-logger's logic.
│   │   └── iter{}_sim_out_1_{compliance value} # An image of the Von-Mises Stress Distribution of the structure.
│   └── ...
├── finite_element_solvers/                             # Implementations of FEM Solvers using basic Python libraries
│   ├── __init__.py                  # Has the function "select_2D_quadrature_solver" to determine an appropriate FEM Solver given the problem to solve.
│   ├── common.py                    # Implements common functions for 2D quadrature solvers
│   ├── four_point_quadrature_plane_stress_composite.py   # Mean for 2D quadrature solver with Lamination Parameter Definition
│   └── four_point_quadrature_plane_stress.py             # Plain 2D quadrature solver.
├── geometry_parameterizations/                           # Geometry Parameterizations
│   ├── __init__.py
│   ├── MMC.py                       # Implements the straight beam definition of the Moving Morphable Components.
├── main_examples/                   # Describe some examples to run problems, wherein the user can define the problem by using the IOH Wrapper classes.
│   └── ...
├── material_parameterizations/      # This directory is to store different material parameterizations (material definitions varying in space)
│   └── ...
├── meshers/      # This directory is to store different material parameterizations (material definitions varying in space)
│   ├── MeshGrid2D.py                # This is an implementation of a simple 2D uniform mesh. It defines an element and node list (without )
│   └── ...
├── utils/                           # Utility functions and helpers
│   ├── Helper_Plots.py              # Within this module you may find plotting definitions for evaluations of designs.
│   ├── Initialization.py            # This is an example of an initialization for isotropic material FEM solver.
│   ├── lame_curves.py               # Module to represent NNC as Lame Curves and compute distance functions in a discrete manner.
│   ├── lame_curves.py               # Module to represent NNC as Lame Curves and compute distance functions in a continuous manner.
│   └── Topology.py                  # Implements the TOpology class, which extends a 2D NumPy array with binary values representing the material spots.
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Files/folders to ignore in git
```

---

## ▶️ Running the Examples

### 🔧 Example: Cantilever Beam (Guo et al., Fig. 9 or Fig. 12)

To reproduce the basic cantilever optimization result:

```bash
python main_example.py
```

Default settings:

* Problem ID: 1 (Cantilever)
* Dimension: 15 (number of MMC design variables)
* Algorithm: CMA-ES (via `cma_es_wrapper`)
* Max evaluations: 1000
* Random Seed: 45

This will:

* Run the optimization loop.
* Log results and positions with IOHAnalyzer.
* Save logs and figures in: `./Figures_Python/Run_29/`

---

## 🤜 Reproducible Results

To reproduce a specific figure from Guo et al.:

* Adjust `dimension`, `problem_id`.
* You can replicate designs like:

  * **Cantilever beam**
  * **Short beam with load at the bottom right end**
  * **MBB Problem**
  * **Michell Truss Problem**
    *(based on Figure 5 from Raponi et.al (2018) paper and Figures 11, 18, 29 of Guo's paper)*

---

## 🧪 Dependencies

Install via:

```bash
pip install -r requirements.txt
```

**Main packages used**:

* `ioh` for logging purposes
* `cma` for CMA-ES
* `numpy` 
* `matplotlib`
* `pyvista` for plotting
* `scipy` for numerical processing and for easy FEM Solver implementation.

---

## 📸 Example Output

Example output figure (`Run_29/Figures/`) shows convergence and stress plots during the optimization process.

---

## Team

* [Elena Raponi](https://www.universiteitleiden.nl/en/staffmembers/elena-raponi#tab-1), *Leiden Institute of Advanced Computer Science*,
* [Iván Olarte Rodríguez](https://www.universiteitleiden.nl/en/staffmembers/ivan-olarte-rodriguez#tab-1), *Leiden Institute of Advanced Computer Science*

## 🧐 Citation

If you use this code, please cite:

```bibtex
@article{guo2016explicit,
  title={Explicit structural topology optimization based on moving morphable components (MMC) with curved skeletons},
  author={Guo, Xu and Zhang, Weisheng and Zhong, Wei},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={310},
  pages={711--748},
  year={2016},
  publisher={Elsevier}
}

@article{raponi_kriging-assisted_2019,
title = {Kriging-assisted topology optimization of crash structures},
volume = {348},
issn = {00457825},
url = {https://linkinghub.elsevier.com/retrieve/pii/S0045782519300726},
doi = {10.1016/j.cma.2019.02.002},
language = {en},
urldate = {2024-08-26},
journal = {Computer Methods in Applied Mechanics and Engineering},
author = {Raponi, Elena and Bujny, Mariusz and Olhofer, Markus and Aulig, Nikola and Boria, Simonetta and Duddeck, Fabian},
month = may,
year = {2019},
pages = {730--752},
}
```
