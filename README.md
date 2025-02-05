# Topology_Optimization
This is a repository for topology optimization.

As of now, the current running code is based on a Concurrent Topology Optimization, wherein just not the topology is optimized, but lamination parameters with Fiber-Steering are optimized given the topology. Furthermore, the only constraints studied are the ones in which the structure is clamped on the left side and there is a point load to the middle-right. Currently, the topology is optimized by using Moving Morphable Components (MMC), an idea from Guo et al. [1], to reduce the dimensionality from the SIMP approach from Bendsoe & Sigmund [4]. For more information on the problem formulation and models behind, the following references are recommended:


[1] X. Guo, W. Zhang, J. Zhang, and J. Yuan, “Explicit structural topology optimization based on moving morphable components (MMC) with curved skeletons,” Computer Methods in Applied Mechanics and Engineering, vol. 310, pp. 711–748, Oct. 2016, doi: 10.1016/j.cma.2016.07.018.

[2] G. Serhat, “Concurrent Lamination and Tapering Optimization of Cantilever Composite Plates under Shear,” Materials, vol. 14, no. 9, p. 2285, Apr. 2021, doi: 10.3390/ma14092285.

[3] E. Raponi, M. Bujny, M. Olhofer, N. Aulig, S. Boria, and F. Duddeck, “Kriging-assisted topology optimization of crash structures,” Computer Methods in Applied Mechanics and Engineering, vol. 348, pp. 730–752, May 2019, doi: 10.1016/j.cma.2019.02.002.

[4] M. P. Bendsøe and O. Sigmund, Topology Optimization. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004. doi: 10.1007/978-3-662-05086-6.

The working code outputs two different plots, which are the deformed structure with the contours of the Von Mises Stress of the material and the distribution of lamination parameters (V1 and V3). A sample of these plots are shown below.
![image](https://github.com/user-attachments/assets/9105270f-19b4-4fd0-9579-9f0ff94f15e3)
![image](https://github.com/user-attachments/assets/02c6ea6e-1cc2-4806-b623-f6f76f9fb223)

## Required Packages
As of now, to run the code the following packages are needed:
- IOH Experimenter (https://github.com/IOHprofiler/IOHexperimenter); Normally installed via `pip install ioh`
- NumPy; IOH Experimenter requires NumPy to be a version below 2.0. So by just installing IOH, then the required NumPy version is installed.
- SciPy; installed via `pip install scipy`. The internal Finite Element Solver uses the linear algebra package as well as sparse package. 
- PyVista; it's for easy visualization purposes. Normally installed via `pip install pyvista`. Take into account that PyVista extends dependencies from Matplotlib, NumPy and VTK. The latter requires that imaging compilers such as Mesa (if you are running Linux) are downloaded.



