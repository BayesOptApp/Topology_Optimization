'''
Import libraries
'''
# OS
import os

# Math
import math

# PyVista
import pyvista

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from matplotlib import ticker, cm

# Numpy
import numpy as np


# -----------------------------------------------------------------------------------------------------
# -------------- CONSTANTS ----------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS:float = 0.2 # -> Linewidth value of element plots
DEFAULT_IMAGE_FILE_FORMAT:str = 'png' #-> Image format to store

DEFAULT_FIGURES_STORAGE_DIRECTORY:str = "Figures_Python"

# -----------------------------------------------------------------------------------------------------
# -------------- HELPER FUNCTIONS ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
 
def quads_to_tris(quad_elements_map:np.ndarray)->np.ndarray:

        '''
        Converts quad elements into tri elements

        ###### Note: In the future, if the matplotlib handles quads, update the code and delete this function

        Inputs:
        - quad_elements_map: array with the mapping of all the nodes belonging to an element
        '''

        tris = [[None for jj in range(4)] for ii in range(2*len(quad_elements_map))]
        for ii in range(len(quad_elements_map)):
                jj = 2*ii
                # Extract the nodes
                n0 = quad_elements_map[ii][1]
                n1 = quad_elements_map[ii][2]
                n2 = quad_elements_map[ii][3]
                n3 = quad_elements_map[ii][4]

                tris[jj][0] = jj
                tris[jj][1] = n0
                tris[jj][2] = n1
                tris[jj][3] = n2

                tris[jj + 1][0] = jj+1
                tris[jj + 1][1] = n2
                tris[jj + 1][2] = n3
                tris[jj + 1][3] = n0

        return np.array(tris,dtype=int,copy=True)

# plots a finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements, axe:plt.Axes,
                  linewidth:float=DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS)->None:
        '''
        Helps to plot the boundaries of the element arrangement given by parameter

        Inputs:
        - nodes_x: array of the x-position of the nodes 
        - nodes_y: array of the y-position of the nodes
        - axe: Axis object to plot the boundaries
        - linewidth: parameter to control the linewidth of the plot
        '''
        for element in elements:
                x = [nodes_x[element[i]] for i in np.arange(start=1,stop=len(element)-1)]
                y = [nodes_y[element[i]] for i in np.arange(start=1,stop=len(element)-1)]
                axe.fill(x, y, edgecolor='black', fill=False,linewidth=linewidth)

# -----------------------------------------------------------------------------------------------------
# ---------------------------MAIN FUNCTIONS -----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

def plotNodalVariables(cost:float,N_static:np.ndarray,element_map:np.ndarray,mat_ind:np.ndarray,
                       nodal_variable:np.ndarray,
                       iterr:int,sample:int,run_:int,
                       plot_elements:bool = True,
                       linewidth:float=DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS,
                       image_file_format:str = DEFAULT_IMAGE_FILE_FORMAT,
                       **kwargs)->None:

        '''
        Plot the contour plot of the nodal variables given by parameter.

        Inputs:
        - cost: value obtained by the cost function
        - N_static: The deformed field of the beam
        - element_map: Array with the node maps of each quad
        - mat_ind: array of booleans defining which elements represent actual material elements
        - nodal_variable: variable to be plotted (defined on nodes)
        - nelx: number of elements in x direction
        - nely: number of elements in y direction
        - iterr: current optimisation iteration
        - sample: sample of evaluation within a population
        - run_: run defined by parameter by the user
        - linewidth: input of the desired linewidth to plot the elements
        - plot_elements: boolean control to show the elements
        - image_file_format: string to determine the format to store the figure
        '''


        # Squeeze (For Precaution)
        mat_ind = np.squeeze(mat_ind)

        # Get final element Map to plot
        final_elem_map:np.ndarray = element_map[mat_ind,:]

        # Get the triangular organization
        triang_org:np.ndarray = quads_to_tris(quad_elements_map=final_elem_map)

        '''
        # Stack the X-Coordinates of each element
        X_patch:np.ndarray = np.hstack((N_static[element_map[mat_ind,1],1].reshape((-1,1)), 
                                N_static[element_map[mat_ind,2],1].reshape((-1,1)),
                                N_static[element_map[mat_ind,3],1].reshape((-1,1)), 
                                N_static[element_map[mat_ind,4],1].reshape((-1,1))))
        
        X_patch_3:np.ndarray = np.hstack((N_static[element_map[mat_ind,1],1].reshape((-1,1)), 
                                N_static[element_map[mat_ind,2],1].reshape((-1,1)),
                                N_static[element_map[mat_ind,3],1].reshape((-1,1)), 
                                N_static[element_map[mat_ind,4],1].reshape((-1,1)),
                                N_static[element_map[mat_ind,1],1].reshape((-1,1))))
        
        # Stack the Y-Coordinates of each element
        Y_patch:np.ndarray = np.hstack((N_static[element_map[mat_ind,1],2].reshape((-1,1)), 
                                N_static[element_map[mat_ind,2],2].reshape((-1,1)),
                                N_static[element_map[mat_ind,3],2].reshape((-1,1)), 
                                N_static[element_map[mat_ind,4],2].reshape((-1,1))))
        
        Y_patch_3:np.ndarray = np.hstack((N_static[element_map[mat_ind,1],2].reshape((-1,1)), 
                                N_static[element_map[mat_ind,2],2].reshape((-1,1)),
                                N_static[element_map[mat_ind,3],2].reshape((-1,1)), 
                                N_static[element_map[mat_ind,4],2].reshape((-1,1)),
                                N_static[element_map[mat_ind,1],2].reshape((-1,1))))
        
        # Stack the values on nodes
        C_patch:np.ndarray = np.hstack((nodal_variable[element_map[mat_ind,1]].reshape((-1,1)), 
                                nodal_variable[element_map[mat_ind,2]].reshape((-1,1)),
                                nodal_variable[element_map[mat_ind,3]].reshape((-1,1)),
                                nodal_variable[element_map[mat_ind,4]].reshape((-1,1))))
        
        C_patch_2:np.ndarray = np.vstack((nodal_variable[element_map[mat_ind,1]].reshape((-1,1)), 
                                nodal_variable[element_map[mat_ind,2]].reshape((-1,1)),
                                nodal_variable[element_map[mat_ind,3]].reshape((-1,1)),
                                nodal_variable[element_map[mat_ind,4]].reshape((-1,1))))
        
        '''

        # Generate the triangulations
        triangulation = tri.Triangulation(N_static[:,1], N_static[:,2], triang_org[:,1:])

        # plot the contours
        fig2, ax2 = plt.subplots()
        ax2.set_aspect('equal')
        tcf = ax2.tricontourf(triangulation,  nodal_variable.flatten(),20,
                                cmap=plt.colormaps['jet'],
                                linewidths=None,
                                locator=ticker.LogLocator(base=10,numticks=20))
        fig2.colorbar(tcf)
        ax2.set_title('Von Mises Stress Contours \n Cost: {0}'.format(cost))

        if plot_elements:
                plot_fem_mesh(nodes_x=N_static[:,1],nodes_y=N_static[:,2],
                                elements=final_elem_map,axe=ax2,
                                linewidth= linewidth)

        
        # Save the figure
        directory_path:str = os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY,
                                          "Run_{0}".format(run_))
        filename:str = "iter{0}_sim_out_{1}_{2}.{3}".format(iterr,
        sample,
        cost,
        image_file_format)

        fullname:str = os.path.join(directory_path,filename)

        try:
                plt.savefig(fullname,format=image_file_format)
        except FileNotFoundError:
                # Check the folder od Figures Exist

                if not os.path.exists(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY)):
                        # Generate the folder
                        os.mkdir(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY))
                        
                if not os.path.exists(directory_path):
                        # Generate the folder
                        os.mkdir(directory_path)
                
                # Try again
                plt.savefig(fullname,format=image_file_format)
        
        # Close the thread of the figure
        plt.close(fig=fig2)


        #plt.show()
        

        #####

        '''
        X = N_static[:,1].reshape((nelx+1,nely+1))
        Y = N_static[:,2].reshape((nelx+1,nely+1))
        Z = nodal_variable.reshape((nelx+1,nely+1))


        fig1, ax1 = plt.subplots()

        codes = [Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,]
        
        patchs:list = []

        
        for itt in range(X_patch_3.shape[0]):
                verts = [(X_patch_3[itt,0], Y_patch_3[itt,0]), 
                        (X_patch_3[itt,1], Y_patch_3[itt,1]), 
                        (X_patch_3[itt,2], Y_patch_3[itt,2]), 
                        (X_patch_3[itt,3], Y_patch_3[itt,3]),
                        (X_patch_3[itt,4], Y_patch_3[itt,4])]
        
        
        
                path = Path(verts,codes)
        
                patch = patches.PathPatch(path, facecolor='none')
                patchs.append(patch)
        
        p = PatchCollection(patchs)
        ax1.add_collection(p)

        ax1.autoscale_view()
        plt.show()
        '''


def plotNodalVariables_pyvista(cost:float,N_static:np.ndarray,element_map:np.ndarray,mat_ind:np.ndarray,
                        nodal_variable:np.ndarray,
                       iterr:int,sample:int,run_:int,
                       plot_elements:bool = True,
                       linewidth:float=DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS,
                       image_file_format:str = DEFAULT_IMAGE_FILE_FORMAT,
                       plot_modifier_dict:dict = None,
                       **kwargs)->None:
        
        '''
        Plot the contour plot of the nodal variables given by parameter by using pyvista.

        Inputs:
        - N_static: The deformed field of the beam
        - element_map: Array with the node maps of each quad
        - mat_ind: array of booleans defining which elements represent actual material elements
        - nelx: number of elements in x direction
        - nely: number of elements in y direction
        - iterr: current optimisation iteration
        - sample: sample of evaluation within a population
        - run_: run defined by parameter by the user
        '''
        
        # Generate an array with the information of vertices
        verts:np.ndarray = np.hstack((N_static[:,1].reshape(-1,1),N_static[:,2].reshape(-1,1)))
        verts:np.ndarray = np.hstack((verts,np.zeros((N_static.shape[0],1))))

        # Squeeze (For Precaution)
        mat_ind = np.squeeze(mat_ind)

        # Get final element Map to plot
        final_elem_map:np.ndarray = element_map[mat_ind,:]

        # Generate an array to point to the Element freedom table
        faces_1:np.ndarray = np.hstack((4*np.ones((final_elem_map.shape[0],1)),
                                          final_elem_map[:,1:5]))

        faces = np.ravel(faces_1)
        faces = faces.astype(int)
        
        # Instantiate a new PyVista Window (for Linux)
        #pyvista.start_xvfb()

        # Generate the mesh
        mesh:pyvista.PolyData = pyvista.PolyData(verts,faces)

        # Repair mesh
        #mesh:pyvista.PolyData = mesh.clean()

        # Extract the dict of the plot modifier
        if plot_modifier_dict is None:
                rotate_bool = False
                rotate_angle = 0.0
        else:
                rotate_bool = plot_modifier_dict.get("rotate",False)
                rotate_angle = plot_modifier_dict.get("rotate_angle",0.0)
        
        if rotate_bool:
                mesh.rotate_z(rotate_angle, inplace=True)
        # Add the data to mesh
        mesh.point_data["Von_Mises_Stress"] = nodal_variable.ravel()

        # Mock Plot
        sargs = dict(
                title = "Von Mises \n Eq. Stress",
                title_font_size=68,
                label_font_size=56,
                n_labels= 5,
                color= "black",
                #fmt="%.4f",
                font_family="arial",
                position_y = 0.1,
                position_x = 0.85,
                vertical = True,
                height = 0.8,
                width = 0.05
                )
        
        p = pyvista.Plotter(border = False,
                            off_screen=True,
                            window_size=[4096, 3072]
                            )
        
        p.enable_image_style()
        
        p.add_mesh(mesh,scalars="Von_Mises_Stress",color="black",
                 scalar_bar_args=sargs,
                 edge_color = "black",
                 show_edges=plot_elements, 
                 interpolate_before_map=True, 
                 cmap="viridis",
                 line_width=linewidth, 
                 log_scale=True)
        
     
        
        p.background_color = 'white'
        #p.add_title('Cost:{0:.4E}'.format(cost), font='arial', color='k', font_size=22)
        #p.show_axes()
        p.show_bounds(color="black",show_zaxis=False, use_2d=True, # font_size = 72,
                      ticks="outside", font_family = "arial", xtitle="", ytitle = "", bold = False)
        
        p.camera.tight(view="xy", adjust_render_window = True,padding=0.6)

        #p.view_xy()


        # Save the figure
        directory_path:str = os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY,
                                          "Run_{0}".format(run_))
        filename:str = "iter{0}_sim_out_{1}_{2}.{3}".format(iterr,
        sample,
        cost,
        image_file_format)

        fullname:str = os.path.join(directory_path,filename)

        try:
                p.screenshot(filename=fullname,transparent_background=False)
        except FileNotFoundError:
                # Check the folder od Figures Exist

                if not os.path.exists(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY)):
                        # Generate the folder
                        os.mkdir(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY))
                        
                if not os.path.exists(directory_path):
                        # Generate the folder
                        os.mkdir(directory_path)
                
                # Try again
                p.screenshot(filename=fullname,transparent_background=False)
        
        # Close the thread of the figure
        p.close()


def plot_LP_Parameters(cost:float,N_static:np.ndarray,element_map:np.ndarray,mat_ind:np.ndarray,
                       iterr:int,sample:int,run_:int,
                       NN:int, NN_l:int, NN_h:int,
                       V1_e:np.ndarray, V3_e:np.ndarray,
                       plot_elements:bool = True,
                       linewidth:float=DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS,
                       image_file_format:str = DEFAULT_IMAGE_FILE_FORMAT,
                       **kwargs)->None:
        
        '''
        Visualise the Lamination parameters 
        Inputs:
        - cost: value obtained by the cost function
        - N_static: The deformed field of the beam
        - element_map: Array with the node maps of each quad
        - mat_ind: array of booleans defining which elements represent actual material elements
        - nelx: number of elements in x direction
        - nely: number of elements in y direction
        - iterr: current optimisation iteration
        - sample: sample of evaluation within a population
        - run_: run defined by parameter by the user
        - NN: Total Number of nodes
        - NN_l: Total number of nodes in x-direction
        - NN_h: Total number of nodes in y-direction
        - V1_e: Array with the elemental V1 parameter
        - V3_e: Array with the elemental V3 parameter
        - linewidth: input of the desired linewidth to plot the elements
        - plot_elements: boolean control to show the elements
        - image_file_format: string to determine the format to store the figure
        '''
        
        # Squeeze (For Precaution)
        mat_ind:np.ndarray = np.squeeze(mat_ind)

        # Total Number of elements
        NE:int = element_map.shape[0]

        V1_n:np.ndarray = np.zeros((NN,1))
        V3_n:np.ndarray = np.zeros((NN,1))

        # Interpolate the element values to nodal values
         # Loop over each element
        for el in range(NE):
                
                #Find the grids of the element
                N1 = element_map[el,1]
                N2 = element_map[el,2]
                N3 = element_map[el,3]
                N4 = element_map[el,4]

                Ne = np.array([N1,N2,N3,N4]).reshape((4,))
                # Apply averaging where multiple elements share the node
                for n in range(4):
                
                        iN = Ne[n]
                        
                        # If the grid is on a corner
                        if ((iN == 0) or (iN == NN_l-1) or (iN == NN-NN_l) or (iN == NN-1)):
                                
                                V1_n[iN]+= V1_e[el]
                                V3_n[iN]+= V3_e[el]
                                
                        # If the grid is on an edge
                        elif (np.fmod(iN,NN_l) == 0) or ((0<iN and iN<NN_l-1)) or (np.fmod(iN+1,NN_l) == 0) or ((NN-NN_l<iN and iN<NN-1)):
                                
                                V1_n[iN]+= V1_e[el]/2
                                V3_n[iN]+= V3_e[el]/2
                                
                        #If the grid is at the interior
                        else:              
                                V1_n[iN]+= V1_e[el]/4
                                V3_n[iN]+= V3_e[el]/4

        # Get final element Map to plot
        final_elem_map:np.ndarray = element_map[mat_ind,:]

        # Get the triangular organization
        triang_org:np.ndarray = quads_to_tris(quad_elements_map=final_elem_map)

        # Generate the triangulations
        triangulation = tri.Triangulation(N_static[:,1], N_static[:,2], triang_org[:,1:])

        
        # 

        # Start the figure object
        # Create two subplots and unpack the output array immediately
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.set_title('V1')
        ax2.set_title('V3')
        
        tcf1 = ax1.tricontourf(triangulation,  V1_n.flatten(),20,
                                cmap="jet",
                                linewidths=None)

        cbar1 = fig.colorbar(tcf1, ax=ax1, location="right")
        cbar1.ax.tick_params(labelsize=5)
        ax1.set_axis_off()

        tcf2 = ax2.tricontourf(triangulation,  V3_n.flatten(),20,
                                cmap="jet",
                                linewidths=None)
        
        cbar2 = fig.colorbar(tcf2, ax=ax2, location="right")
        cbar2.ax.tick_params(labelsize=5)
        ax2.set_axis_off()


        if plot_elements:
                plot_fem_mesh(nodes_x=N_static[:,1],nodes_y=N_static[:,2],
                                elements=final_elem_map,axe=ax1,
                                linewidth= linewidth)
                
                plot_fem_mesh(nodes_x=N_static[:,1],nodes_y=N_static[:,2],
                                elements=final_elem_map,axe=ax2,
                                linewidth= linewidth)
        

        # Save the figure
        directory_path:str = os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY,
                                          "Run_{0}".format(run_))
        filename:str = "iter{0}_LP_distr_{1}_{2}.{3}".format(iterr,
        sample,
        cost,
        image_file_format)

        fullname:str = os.path.join(directory_path,filename)

        try:
                plt.savefig(fullname,format=image_file_format)
        except FileNotFoundError:
                # Check the folder od Figures Exist

                if not os.path.exists(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY)):
                        # Generate the folder
                        os.mkdir(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY))
                        
                if not os.path.exists(directory_path):
                        # Generate the folder
                        os.mkdir(directory_path)
                
                # Try again
                plt.savefig(fullname,format=image_file_format)


        # Close the thread of the figure
        plt.close(fig=fig)
        
        
        #ax1.plot(x, y)
        #ax1.set_title('Sharing Y axis')
        #ax2.scatter(x, y)

def plot_LP_Parameters_pyvista(cost:float,N_static:np.ndarray,element_map:np.ndarray,mat_ind:np.ndarray,
                       iterr:int,sample:int,run_:int,
                       V1_e:np.ndarray, V3_e:np.ndarray,
                       plot_elements:bool = True,
                       linewidth:float=DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS,
                       image_file_format:str = DEFAULT_IMAGE_FILE_FORMAT,
                       plot_modifier_dict:dict = None,
                       **kwargs)->None:
        
        '''
        Visualise the Lamination parameters 
        Inputs:
        - cost: value obtained by the cost function
        - N_static: The deformed field of the beam
        - element_map: Array with the node maps of each quad
        - mat_ind: array of booleans defining which elements represent actual material elements
        - nelx: number of elements in x direction
        - nely: number of elements in y direction
        - iterr: current optimisation iteration
        - sample: sample of evaluation within a population
        - run_: run defined by parameter by the user
        - V1_e: Array with the elemental V1 parameter
        - V3_e: Array with the elemental V3 parameter
        - linewidth: input of the desired linewidth to plot the elements
        - plot_elements: boolean control to show the elements
        - image_file_format: string to determine the format to store the figure
        '''

        # Generate an array with the information of vertices
        verts:np.ndarray = np.hstack((N_static[:,1].reshape(-1,1),N_static[:,2].reshape(-1,1)))
        verts:np.ndarray = np.hstack((verts,np.zeros((N_static.shape[0],1))))

        # Squeeze (For Precaution)
        mat_ind = np.squeeze(mat_ind)

        # Get final element Map to plot
        final_elem_map:np.ndarray = element_map[mat_ind,:]

        # Generate an array to point to the Element freedom table
        faces_1:np.ndarray = np.hstack((4*np.ones((final_elem_map.shape[0],1)),
                                          final_elem_map[:,1:5]))

        faces = np.ravel(faces_1)
        faces = faces.astype(int)

        # Generate a new PyVista window (for Linux)
        #pyvista.start_xvfb()

        # Generate the mesh
        mesh:pyvista.PolyData = pyvista.PolyData(verts,faces)


        # Repair mesh
        #mesh:pyvista.PolyData = mesh.clean()

        # Extract the dict of the plot modifier
        if plot_modifier_dict is None:
                rotate_bool = False
                rotate_angle = 0.0
        else:
                rotate_bool = plot_modifier_dict.get("rotate",False)
                rotate_angle = plot_modifier_dict.get("rotate_angle",0.0)
        
        if rotate_bool:
                mesh.rotate_z(rotate_angle, inplace=True)

        # Add the data to the mesh
        mesh.cell_data["V1"] = V1_e[mat_ind].ravel()
        mesh.cell_data["V3"] = V3_e[mat_ind].ravel()

        #mesh:pyvista.PolyData = mesh.cell_data_to_point_data()

        # Mock Plot

        p = pyvista.Plotter(border = False,
                            off_screen=True,
                            shape=(1, 2),
                            window_size=[8192, 6144]
                            )
        
        p.enable_image_style()
        
        p.subplot(0,0)

       
        sargs1 = dict(title = "V1",
                        title_font_size=72,
                        label_font_size=56,
                        n_labels=5,
                        color= "black",
                        #fmt="%.4f",
                        font_family="arial",
                        position_x = 0.20,
                        position_y = 0.75,
                        height = 0.05,
                        width = 0.6,
                        vertical = False
                        )


        
        p.add_mesh(mesh,copy_mesh=False,scalars="V1",edge_color="black",
                   color = "green",scalar_bar_args = sargs1,
                 show_edges=plot_elements, 
                 interpolate_before_map=True, cmap="jet",
                 line_width=linewidth, clim = [-1.0,1.0],
                 log_scale=False)
        


       
        p.background_color = 'white'
        #p.camera_position ="xy"
        p.camera.tight(view="xy", adjust_render_window = True,padding=0.4)
        #p.add_title('Cost:{0:.4E}'.format(cost), font='arial', color='k', font_size=22)
        #p.show_axes()
        #p.show_bounds(color="black",show_zaxis=False,font_size=20)

        # get the bounds of the first subplot 


        p.subplot(0,1)


        sargs2 = dict(title = "V3",
                        title_font_size=72,
                        label_font_size=56,
                        n_labels=5,
                        color= "black",
                        #fmt="%.4f",
                        font_family="arial",
                        position_x = 0.20,
                        position_y = 0.75,
                        width = 0.6,
                        height = 0.05,
                        vertical = False
                        )


        p.add_mesh(mesh,copy_mesh=True,scalars="V3",color="black",edge_color="black",
                 show_edges=plot_elements, scalar_bar_args = sargs2,
                 interpolate_before_map=True, cmap="jet",
                 line_width=linewidth, clim = [-1.0,1.0],
                 log_scale=False)
        
      

        p.background_color = 'white'
        #p.add_title('Cost:{0:.4E}'.format(cost), font='arial', color='k', font_size=22)

        p.camera.tight(view="xy", adjust_render_window = True,padding=0.4)
        
        # Save the figure
        directory_path:str = os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY,
                                          "Run_{0}".format(run_))
        filename:str = "iter{0}_LP_dist_{1}_{2}.{3}".format(iterr,
        sample,
        cost,
        image_file_format)

        fullname:str = os.path.join(directory_path,filename)

   
        try:
                p.screenshot(filename=fullname,transparent_background=False)
        except FileNotFoundError:
                # Check the folder od Figures Exist

                if not os.path.exists(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY)):
                        # Generate the folder
                        os.mkdir(os.path.join(os.getcwd(),
                                          DEFAULT_FIGURES_STORAGE_DIRECTORY))
                        
                if not os.path.exists(directory_path):
                        # Generate the folder
                        os.mkdir(directory_path)
                
                # Try again
                p.screenshot(filename=fullname,transparent_background=False)
        
        # Close the thread of the figure
        p.close()


def plot_LP_Parameters_pyvista_2(N_static:np.ndarray,
                                 element_map:np.ndarray,
                                 mat_ind:np.ndarray,
                                 V1_e:np.ndarray, 
                                 V3_e:np.ndarray,
                                 plot_elements:bool = True,
                                 linewidth:float=DEFAULT_PLOT_LINEWIDTH_ELEMENT_PLOTS,
                                 plot_modifier_dict:dict = None,
                                **kwargs)->pyvista.Plotter:
        
        r'''
        Visualise the Lamination parameters
        This function is used to plot the Lamination parameters V1 and V3 in a pyvista window.

        Args
        ------------------------
        - N_static: The deformed field of the beam
        - element_map: Array with the node maps of each quad
        - mat_ind: array of booleans defining which elements represent actual material elements
        - V1_e: Array with the elemental V1 parameter
        - V3_e: Array with the elemental V3 parameter
        - linewidth: input of the desired linewidth to plot the elements
        - plot_elements: boolean control to show the elements
        - plot_modifier_dict: dictionary with the plot modifiers, e.g. rotation angle

        Returns:
        ------------------------
        p: pyvista.Plotter object with the mesh of the Lamination parameters
        '''

        # Generate an array with the information of vertices
        verts:np.ndarray = np.hstack((N_static[:,1].reshape(-1,1),N_static[:,2].reshape(-1,1)))
        verts:np.ndarray = np.hstack((verts,np.zeros((N_static.shape[0],1))))

        # Squeeze (For Precaution)
        mat_ind = np.squeeze(mat_ind)

        # Get final element Map to plot
        final_elem_map:np.ndarray = element_map[mat_ind,:]

        # Generate an array to point to the Element freedom table
        faces_1:np.ndarray = np.hstack((4*np.ones((final_elem_map.shape[0],1)),
                                          final_elem_map[:,1:5]))

        faces = np.ravel(faces_1)
        faces = faces.astype(int)

        # Generate a new PyVista window (for Linux)
        #pyvista.start_xvfb()

        # Generate the mesh
        mesh:pyvista.PolyData = pyvista.PolyData(verts,faces)


        # Repair mesh
        #mesh:pyvista.PolyData = mesh.clean()

        # Extract the dict of the plot modifier
        if plot_modifier_dict is None:
                rotate_bool = False
                rotate_angle = 0.0
        else:
                rotate_bool = plot_modifier_dict.get("rotate",False)
                rotate_angle = plot_modifier_dict.get("rotate_angle",0.0)
        
        if rotate_bool:
                mesh.rotate_z(rotate_angle, inplace=True)

        # Add the data to the mesh
        mesh.cell_data["V1"] = V1_e[mat_ind].ravel()
        mesh.cell_data["V3"] = V3_e[mat_ind].ravel()

        #mesh:pyvista.PolyData = mesh.cell_data_to_point_data()

        # Mock Plot

        p = pyvista.Plotter(border = False,
                            off_screen=True,
                            shape=(1, 2),
                            window_size=[8192, 6144]
                            )
        
        p.enable_image_style()
        
        p.subplot(0,0)

       
        sargs1 = dict(title = "V1",
                        title_font_size=72,
                        label_font_size=56,
                        n_labels=5,
                        color= "black",
                        #fmt="%.4f",
                        font_family="arial",
                        position_x = 0.20,
                        position_y = 0.75,
                        height = 0.05,
                        width = 0.6,
                        vertical = False
                        )


        
        p.add_mesh(mesh,copy_mesh=False,scalars="V1",edge_color="black",
                   color = "green",scalar_bar_args = sargs1,
                 show_edges=plot_elements, 
                 interpolate_before_map=True, cmap="jet",
                 line_width=linewidth, clim = [-1.0,1.0],
                 log_scale=False)
        


       
        p.background_color = 'white'
        #p.camera_position ="xy"
        p.camera.tight(view="xy", adjust_render_window = True,padding=0.4)
        #p.add_title('Cost:{0:.4E}'.format(cost), font='arial', color='k', font_size=22)
        #p.show_axes()
        #p.show_bounds(color="black",show_zaxis=False,font_size=20)

        # get the bounds of the first subplot 


        p.subplot(0,1)


        sargs2 = dict(title = "V3",
                        title_font_size=72,
                        label_font_size=56,
                        n_labels=5,
                        color= "black",
                        #fmt="%.4f",
                        font_family="arial",
                        position_x = 0.20,
                        position_y = 0.75,
                        width = 0.6,
                        height = 0.05,
                        vertical = False
                        )


        p.add_mesh(mesh,copy_mesh=True,scalars="V3",color="black",edge_color="black",
                 show_edges=plot_elements, scalar_bar_args = sargs2,
                 interpolate_before_map=True, cmap="jet",
                 line_width=linewidth, clim = [-1.0,1.0],
                 log_scale=False)
        
      

        p.background_color = 'white'
        #p.add_title('Cost:{0:.4E}'.format(cost), font='arial', color='k', font_size=22)

        p.camera.tight(view="xy", adjust_render_window = True,padding=0.4)
        
        # Save the figure
        # directory_path:str = os.path.join(os.getcwd(),
        #                                   DEFAULT_FIGURES_STORAGE_DIRECTORY,
        #                                   "Run_{0}".format(run_))
        # filename:str = "iter{0}_LP_dist_{1}_{2}.{3}".format(iterr,
        # sample,
        # cost,
        # image_file_format)

        #fullname:str = os.path.join(directory_path,filename)

   
        # try:
        #         p.screenshot(filename=fullname,transparent_background=False)
        # except FileNotFoundError:
        #         # Check the folder od Figures Exist

        #         if not os.path.exists(os.path.join(os.getcwd(),
        #                                   DEFAULT_FIGURES_STORAGE_DIRECTORY)):
        #                 # Generate the folder
        #                 os.mkdir(os.path.join(os.getcwd(),
        #                                   DEFAULT_FIGURES_STORAGE_DIRECTORY))
                        
        #         if not os.path.exists(directory_path):
        #                 # Generate the folder
        #                 os.mkdir(directory_path)
                
        #         # Try again
        #         p.screenshot(filename=fullname,transparent_background=False)
        
        # Close the thread of the figure
        #p.close()
        return p
       