import numpy as np
from MPLC.beam import Beam, Set_Beams
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import fft2, fftshift,ifft2
import os



class System:
    def __init__(self,**kwargs) -> None:
        
        # Define the system
        self.define_system(**kwargs)

        # Collects the set of beams in the system
        self.set_beams = {}

    def define_system(self,
                      Nx: int = 256, 
                      Ny: int  =256, 
                      size: float = 8e-6, 
                      zoom_out:list[int] = [2, 2], 
                      num_planes: int = 7, 
                      planeDist: float = 25.14e-3, 
                      firstPlaneDist: float = 20e-3,
                      iterations: int = 100,
                      kSpaceFilter: int =1000,
                      symmetricMasks = True,
                      maskOffset = None,
                      w0: float = 60e-6,
                      w1: float = 400e-6,
                      lam: float = 1565e-9,
                      num_modes: int = 5
                     ) -> None:
        # ----------------------------------------
        #
        # Initializing variables of the MPLC system
        #
        # Inputs:
        #
        # Nx = Number of elements in the x-direction of the grid/mask 
        # Ny = Number of elements in the y-direction of the grid/mask
        # size = Spacing of each element/pixel size
        # zoom_out = A factor that multiplies the space dimension of the grid
        # num_planes = Number of masks/mirrors in the MPLC system
        # planeDist = Distance between masks/mirrors
        # firstPlaneDist = Distance of the first mask/mirror to the laser
        # iterations = Number of iterations for the wavefront matching algorithm
        # kSpaceFilter = Filters beams propagating with an angle less than kSpaceFilter*size
        # symmetricMasks = Force the masks to be symmetric
        # maskOffset = Offset added to the mask, removes the possibility to phase match very low intensity fields.
        # w0 = Mode-field diameter of the entering beams
        # w1 = Mode-field diameter of the exiting beams
        # lam = wavelength of the beams
        # num_modes = Number of beam modes 
        #
        # ----------------------------------------

        # Setting instance variables
        self.Nx = Nx
        self.Ny = Ny
        self.size = size
        x = np.linspace(- Nx * size * zoom_out[0] / 2.0, Nx * size * zoom_out[0] / 2.0, Nx )
        y = np.linspace(- Ny * size * zoom_out[1] / 2.0, Ny * size * zoom_out[1] / 2.0, Ny )
        self.x = x + ( x[1] - x[0] ) / 2.0
        self.y = y + ( y[1] - y[0] ) / 2.0
        self.w0 = w0
        self.w1 = w1
        self.lam = lam

        self.kSpaceFilter = kSpaceFilter
        self.num_iterations = iterations
        self.num_modes= num_modes

        # Number of beams is dependent of the number of modes
        self.num_spots = np.sum(np.arange(1,num_modes+1))
        self.num_planes = num_planes
        self.planeDist = planeDist
        self.symmetricMasks = symmetricMasks
        self.firstPlaneDist = firstPlaneDist
        self.X, self.Y = np.meshgrid(self.x,self.y)

        # Radius from polar coordinates
        self.R = np.sqrt(self.X**2 + self.Y**2,dtype="float32")



        if maskOffset is None:
            self.maskOffset = np.sqrt(1e-3/(1.0*self.Nx*self.Ny*self.num_spots))


    


    def create_Gaussians(self, 
                         positions: list[float] | str | None = None,
                         num_modes: int | None = None, 
                         w0 : float | None  = None,
                         lam : float | None  = None,
                         firstPlaneDist: float | None = None,
                         **kwargs) -> Set_Beams:
        
        # ----------------------------------------
        #
        # Generation of a set of Gaussian beams
        #
        # Inputs:
        #
        # positions = Defining the positions of the beams
        # num_modes = Number of beam modes 
        # w0 = Mode-field diameter of the Gaussian beams
        # lam = wavelength of the beams
        # firstPlaneDist = Distance of the first mask/mirror to the laser
        #
        # ----------------------------------------

        if firstPlaneDist is None:
            firstPlaneDist = self.firstPlaneDist

        # Defining the Z-grid at a distance defined by firstPlaneDist
        Z = np.ones((self.Nx,self.Ny), dtype="float32") * firstPlaneDist

        if w0 is None:
            w0 = self.w0
        if lam is None:
            lam = self.lam
        if num_modes is None:
            num_modes = self.num_modes

        
        num_beams = len(self.set_beams)

        return Set_Beams(id = num_beams, type="gaussian", X = self.X, Y = self.Y, w = w0, lam = lam, num_modes= num_modes, positions= positions, Z = Z, **kwargs)
        
    def create_HermiteGaussian(self,
                               positions: list[float] | str | None = None,
                               num_modes: int | None = None, 
                               w1 : float | None  = None,
                               lam : float | None  = None,
                               plot : bool = False,
                               rotation: float | None = None,
                               **kwargs) -> Set_Beams:
        
        # ----------------------------------------
        #
        # Generation of a set of Hermite Gaussian beams
        #
        # Inputs:
        #
        # positions = Defining the positions of the beams
        # num_modes = Number of beam modes 
        # w1 = Mode-field diameter of the Hermite Gaussian beams
        # lam = wavelength of the beams
        # plot = If the propose of the set of beams are for plotting
        # rotation = rotation of the beams
        #
        # ----------------------------------------

        if w1 is None:
            w1 = self.w1
        if lam is None:
            lam = self.lam
        if num_modes is None:
            num_modes = self.num_modes


        # Default rotation pi/4
        Phi = np.arctan2(self.Y,self.X,dtype="float32")

        if rotation is None:
            X = self.R * np.cos(Phi- np.pi/4.0)
            Y = self.R * np.sin(Phi- np.pi/4.0)
        else:
            X = self.R * np.cos(Phi- rotation)
            Y = self.R * np.sin(Phi- rotation)


        num_beams = len(self.set_beams)

        return Set_Beams(id = num_beams, type="hermite", X = X, Y = Y, w = w1, lam = lam, num_modes=num_modes, positions= positions, plot = plot, **kwargs )
    

    def set_fields(self, fieldIn: Set_Beams, fieldOut: Set_Beams) -> None:
        # ----------------------------------------
        #
        # Set the masks and fields in the systems
        #
        # Inputs:
        #
        # fieldIn = Fields that will propagate forward in the system
        # fieldOut = Fields that will propagate backward in the system
        #
        # ----------------------------------------
        
        self.MASKS = np.ones((self.num_planes, self.Nx, self.Ny), dtype="csingle")
        

        # Initialize fields
        self.FIELDS = np.zeros((2,self.num_planes, self.num_spots, self.Nx, self.Ny),dtype="csingle")
        for i, field in enumerate(fieldIn.beams):
            self.FIELDS[0,0,i,:,:] = np.copy(field.result)
        for i, field in enumerate(fieldOut.beams):
            self.FIELDS[1,-1,i,:,:] = np.copy(field.result)
        
        # Define the equation that will interfer with the field propagating between masks ( vaccum )
        self.H0 = Beam.freeSpace(self.Nx, self.Ny,self.X, self.Y, self.planeDist, self.lam).result
        maxR = np.max(self.R)
        self.H = self.H0 * ( self.R < ( self.kSpaceFilter * maxR))
        
        # Set the first forward fields in the different masks with interferance by the propagation in the vaccum
        h = self.H
        direction = 0
        for id_plane in range(self.num_planes-1):
            MASK = np.exp(-1j* np.angle(np.copy(self.MASKS[id_plane,:,:])))
            for id_spot in range(self.num_spots):
                FIELD = np.copy(self.FIELDS[direction,id_plane, id_spot, :,:])
                self.FIELDS[direction,id_plane + 1, id_spot, :,:] = self.update_FIELD(FIELD, MASK, h)
            
        # Set the first backward fields in the different masks with interferance by the propagation in the vaccum
        h = np.conjugate(self.H)
        direction = 1
        for id_plane in range(self.num_planes-1,0,-1):
            MASK = np.exp(1j* np.angle(np.copy(self.MASKS[id_plane,:,:])))
            for id_spot in range(self.num_spots):
                FIELD = np.copy(self.FIELDS[direction,id_plane, id_spot, :,:])
                self.FIELDS[direction,id_plane - 1, id_spot, :,:] = self.update_FIELD(FIELD, MASK, h)
            

    def start(self, fieldIn: Set_Beams | None = None, fieldOut: Set_Beams | None = None) -> None:

        # ----------------------------------------
        #
        # Beginning of the wavefront matching algorithm to define masks that change Gaussians to Hermite-Gaussian beams
        #
        # Inputs:
        #
        # fieldIn = Fields that will propagate forward in the system
        # fieldOut = Fields that will propagate backward in the system
        #
        # ----------------------------------------
        
        if fieldIn is not None and fieldOut is not None:
            self.set_fields(fieldIn, fieldOut)

        
        for i in range(self.num_iterations):

            # Forward direction
            h = self.H
            direction = 0
            for id_plane in range(self.num_planes-1):

                # Update of the masks
                MASK_in = np.copy(self.MASKS[id_plane,:,:])
                self.MASKS[id_plane,:,:] = self.update_MASK(id_plane, MASK_in )
                
                # Update of the fields 
                MASK_in = np.exp(-1j * np.angle(np.copy(self.MASKS[id_plane,:,:])))
                for id_spot in range(self.num_spots):
                    FIELD = np.copy(self.FIELDS[direction,id_plane, id_spot, :,:])
                    self.FIELDS[direction,id_plane + 1, id_spot, :,:] = self.update_FIELD(FIELD, MASK_in, h)
                
            # Backward direction
            h = np.conjugate(self.H)
            direction = 1
            for id_plane in range(self.num_planes-1,0,-1):
                # Update of the masks
                MASK_in = np.copy(self.MASKS[id_plane,:,:])
                self.MASKS[id_plane,:,:] = self.update_MASK(id_plane, MASK_in )

                # Update of the fields
                MASK_in = np.exp(1j * np.angle(np.copy(self.MASKS[id_plane,:,:])))
                for id_spot in range(self.num_spots):
                    FIELD = np.copy(self.FIELDS[direction,id_plane, id_spot, :,:])
                    self.FIELDS[direction,id_plane - 1, id_spot, :,:] = self.update_FIELD(FIELD, MASK_in, h)

        # Last update of the fields with the converged masks
        h = self.H0
        direction = 0
        self.Total_FIELDS = np.zeros((self.num_planes,self.Nx, self.Ny), dtype="float32")
        #self.Total_FIELDS = np.zeros((self.num_planes,self.Nx, self.Ny), dtype="float32")
        for id_plane in range(self.num_planes-1):
            MASK_in = np.exp(-1j * np.angle(np.copy(self.MASKS[id_plane,:,:])))

            for id_spot in range(self.num_spots):
                FIELD = np.copy(self.FIELDS[direction,id_plane, id_spot, :,:])
                self.FIELDS[direction,id_plane + 1, id_spot, :,:] = self.update_FIELD(FIELD, MASK_in, h)
            
                self.Total_FIELDS[id_plane,:,:] += np.abs(self.FIELDS[direction,id_plane, id_spot, :,:])**2

        id_plane = self.num_planes-1
        for id_spot in range(self.num_spots):
            self.Total_FIELDS[id_plane,:,:] += np.abs(self.FIELDS[direction,id_plane, id_spot, :,:])**2
    def update_FIELD(self, field, mask, h):
        # ----------------------------------------
        #
        # Update of the field during the vacumm propagation
        #
        # Inputs:
        #
        # field = Field
        # mask = Mask
        # h = Transfer function
        #
        # ----------------------------------------

        new_field = field * mask
        field_fft = fftshift(fft2(fftshift(new_field)))
        field_fft = field_fft * h
        new_field = fftshift(ifft2(fftshift(field_fft)))
        



        return new_field

    def update_MASK(self,id_plane, old_Mask):
        # ----------------------------------------
        #
        # Update of the mask by difference of phase between forward and backward fields
        #
        # Inputs:
        #
        # id_plane = Index of the plane
        # old_Mask = Mask previous to the update
        #
        # ----------------------------------------

        MASK_old = np.exp(1j * np.angle(old_Mask))
        new_MASK = np.zeros((self.Nx,self.Ny),dtype="csingle")
        
        for id_spot in range(self.num_spots):
            
            # Overlap of the fields normalized by their power
            fF = np.copy(self.FIELDS[0,id_plane,id_spot,:,:])
            bF = np.conjugate(np.copy(self.FIELDS[1,id_plane,id_spot,:,:]))
            pwr_fF = np.sum(np.abs(fF)**2)
            pwr_bF = np.sum(np.abs(bF)**2)
            Overlap_fields = fF * bF / np.sqrt(pwr_bF * pwr_fF)
            
            # Overlap of the mask
            Overlap_masks = np.sum(Overlap_fields * np.conjugate(MASK_old))
  
            # New mask summed over each beam
            new_MASK += Overlap_fields * np.exp(-1j * np.angle(Overlap_masks))

        # If symmetry is forced
        if self.symmetricMasks:
            new_MASK = ( new_MASK + new_MASK[::-1,:])/2.0
    
        # Add mask offset
        new_MASK += self.maskOffset

        return new_MASK



    def get_couplingMatrix(self):
        # ----------------------------------------
        #
        # At the end of convergence of masks, obtain the single values of the transfer matrix and calculate the insertion losse and the mode-dependent loss
        #
        # ----------------------------------------

        self.couplingMatrix = np.zeros((self.num_spots, self.num_spots), dtype="csingle")

        for id_spot1 in range(self.num_spots):
            direction = 0
            fIn = np.conjugate(np.copy(self.FIELDS[direction, self.num_planes-1, id_spot1,:,:])) * np.exp(1j * np.angle(np.copy(self.MASKS[self.num_planes - 1,:,:])))
            direction = 1
            for id_spot2 in range(self.num_spots):
                fOut = np.copy(self.FIELDS[direction,self.num_planes-1, id_spot2, :, :])
                self.couplingMatrix[id_spot1,id_spot2] = np.sum(fIn * fOut)


        [U, S, Vh] = np.linalg.svd(self.couplingMatrix)

        s = np.diag(S)**2

        # Set average weights at 1
        w_s = np.zeros(s.shape)
        for i in range(len(s)):
            w_s[i,i]=1

        IL = 10 * np.log10(np.average(s,weights=w_s))
        MDL = 10 * np.log10(s[-1,-1]/s[0,0])

        return s, IL, MDL
    

    def plot_fields(self, type_plot = "both", separate = True, num_planes: list[int] | int | None = None, join_x: bool = False, join_y: bool = False, save: str = "", **kwargs):
        # ----------------------------------------
        #
        # Update of the mask by difference of phase between forward and backward fields
        #
        # Inputs:
        #
        # type_plot = ("both", "fields", "masks") choose the type of plot
        # separate = True: each plane has its own figure; False: or not
        # num_planes = List of planes to plot; Number of planes to plot including the first and the last; None: all of them
        # join_x = The x-axis is the same for fields and masks ( for type_plot=="both" )
        # join_y = The y-axis is the same ( for separate = False )
        # save = Chose the directory where to save the plots.
        # 
        # kwargs include:
        #
        # fontsize = Size of the font in the plots
        #
        # ----------------------------------------

        # Do not plot when creating figures
        plt.ioff()

        if not os.path.exists(save):
            os.mkdir(save)
        
        if "fontsize" in kwargs:
            fontsize = kwargs["fontsize"]
        else:
            fontsize=14
        plt.rc('font', size=fontsize)

        if type_plot == "both":
            sub_num = 2
        else:
            sub_num = 1
            
        # Configure the number of planes to plot
        if num_planes is None:
            planes = range(self.num_planes)
        elif type(num_planes)  == list:
            planes = num_planes
        elif type(num_planes) == int:
            if num_planes != 1:
                if num_planes > self.num_planes:
                    planes = range(self.num_planes)
                else:
                    spacing = int(self.num_planes/(num_planes-1))
                    planes = [ i for i in range(0,self.num_planes,spacing)]
                    if spacing == self.num_planes:
                        planes.append(self.num_planes-1)
            else:
                planes = [0]
            

            
        # Configure the figures and axes to plot
        figures = []
        axes = []
        if separate == True:
            for i in range(len(planes)):
                fig_temp, ax_temp = plt.subplots(sub_num,figsize=(8,8))
                figures.append(fig_temp)
                axes.append(ax_temp)
        else:
            fig_temp, ax_temp = plt.subplots(sub_num, len(planes),figsize=(18,4))
            figures.append(fig_temp)
            axes=ax_temp
        
        
        for i,fig in enumerate(figures):

            # Set the margin for the different type of plots 
            xlmargem = 0.12
            xrmargem = 0.94
            ybmargem = 0.08
            ytmargem= 0.95
            wspace = 0
            hspace = 0
            toptext = "Intensity of the fields\ny(m)"
            bottomtext = "Phase mask\n(ym)"
            if join_y and not join_x:
                wspace = 0
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,wspace = wspace)
            elif join_x and not join_y:
                hspace = 0
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,hspace = hspace)
            elif join_x and join_y:
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,wspace = wspace, hspace=hspace)
            else:
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem)
            if (type_plot == "fields" or type_plot == "masks") and separate:
                xlmargem = 0.15
                xrmargem = 0.94
                ybmargem = 0.08
                ytmargem= 0.95
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem)

            if (type_plot == "fields" or type_plot == "masks") and not separate:
                xlmargem = 0.06
                xrmargem = 0.98
                ybmargem = 0.15
                ytmargem= 0.86
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem)


            if type_plot == "both" and not separate and join_x:
                xlmargem = 0.07
                ybmargem = 0.17
                xrmargem = 0.96
                ytmargem = 0.86
                hspace = 0
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,hspace = hspace)
                toptext = "Intensity\nof the fields\ny(m)"
                bottomtext = "Phase mask\n(ym)"
            if type_plot == "both" and not separate and not join_x:# and not join_y:
                xlmargem = 0.07
                ybmargem = 0.15
                xrmargem = 0.97
                ytmargem = 0.86
                hspace = 0
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,hspace = hspace)
                toptext = "Intensity\nof the fields\ny(m)"
                bottomtext = "Phase mask\n(ym)"
            if type_plot=="both" and not join_x:
                if separate:
                    fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,hspace = 0.2)
                else:
                    fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,hspace = 0.5)

            if not separate and not join_y:
                fig.subplots_adjust(xlmargem,ybmargem,xrmargem,ytmargem,wspace=0.4)

            if type_plot=="both":
                if separate:
                    f = axes
                else:
                    f = np.transpose(axes)
            else:
                f = axes
            for i_ax,ax in enumerate(f):

                j = planes[i_ax]
                
                # Plot if the type is just fields
                if type_plot == "fields":
                    ax.set_title("Plane "+str(planes[j]+1))
   
                    ax.pcolormesh(self.X,self.Y, self.Total_FIELDS[j])

                # Plot if the type is just masks
                elif type_plot == "masks":
                    ax.set_title("Plane "+str(planes[j]+1))
                    min_mask = np.min(np.angle(self.MASKS[j]))
                    max_mask = np.max(np.angle(self.MASKS[j]))
                    im = ax.pcolormesh(self.X, self.Y, np.angle(self.MASKS[j]),cmap=mpl.colormaps["hsv"],vmin=-np.pi,vmax=np.pi)
                    divider = make_axes_locatable(ax)
                    if not join_y or i_ax == len(f)-1:
                        cax = divider.append_axes('right', size='2%', pad=-0.05)
                        
                        cbar = fig.colorbar(mappable=im , cax= cax)
                        cbar.set_ticks([-np.pi,0,np.pi])
                        
                        cbar.set_ticklabels([r"$-\pi$","0",r"$\pi$"])
                elif type_plot == "both":
                    
                    ax1 = ax[0]
                    ax1.pcolormesh(self.X,self.Y, self.Total_FIELDS[j])

                    ax1.set_title("Plane "+str(j+1))

                    # Design for the cases of joint x-/y- axis
                    if i_ax ==0 or separate:
                        ax1.set_ylabel(toptext)
                    if separate or (not join_x and not join_y):
                            if join_x:
                                ax1.ticklabel_format(axis='y', scilimits=[-1, 1])
                            else:
                                ax1.ticklabel_format(axis='both', scilimits=[-1, 1])

                    if join_x :
                        ax1.set_xticklabels([])
                        ax1.set_xticks([])
                        ax1.ticklabel_format(axis='y', scilimits=[-1, 1])
                        tick = ax1.yaxis.get_majorticklabels()[1]
                        tick.set_verticalalignment('bottom')
                    
                    if join_y and not separate:
                        if i_ax !=0:
                            ax1.set_yticklabels([])
                            ax1.set_yticks([])
                        else:
                            ax1.ticklabel_format(axis='y', scilimits=[-1, 1])
                        if not join_x:
                            ax1.ticklabel_format(axis='x', scilimits=[-1, 1])

                            if i_ax ==0:
                                tick = ax1.xaxis.get_majorticklabels()[-2]
                                tick.set_horizontalalignment('right')
                            elif i_ax == len(f)-1:
                                tick = ax1.xaxis.get_majorticklabels()[1]
                                tick.set_horizontalalignment('left')
                            else:
                                tick = ax1.xaxis.get_majorticklabels()[-2]
                                tick.set_horizontalalignment('right')
                                tick = ax1.xaxis.get_majorticklabels()[1]
                                tick.set_horizontalalignment('left')

                    ax2 = ax[1]
                    im = ax2.pcolormesh(self.X, self.Y, np.angle(self.MASKS[j]),cmap=mpl.colormaps["hsv"],vmin=-np.pi,vmax=np.pi)
                    ax2.set_xlabel("x (m)")

                    # Design for the cases of joint x-/y- axis
                    if i_ax ==0 or separate:
                        ax2.set_ylabel(bottomtext)
                    if separate or (not join_x and not join_y):
                        ax2.ticklabel_format(axis='both', scilimits=[-1, 1])

                    if join_x :
                        ax2.ticklabel_format(axis='both', scilimits=[-1, 1])
                        tick = ax2.yaxis.get_majorticklabels()[-2]
                        tick.set_verticalalignment('top')
                        if not separate:
                            ax2.get_xaxis().get_offset_text().set_position((0.0,0.5))


                    if join_y and not separate:
                        if i_ax !=0:
                            ax2.set_yticklabels([])
                            ax2.set_yticks([])
                            ax2.ticklabel_format(axis='x', scilimits=[-1, 1])
                        else:
                            ax2.ticklabel_format(axis='both', scilimits=[-1, 1])
                            

                        if i_ax ==0:
                            tick = ax2.xaxis.get_majorticklabels()[-2]
                            tick.set_horizontalalignment('right')
                        elif i_ax == len(f)-1:
                            tick = ax2.xaxis.get_majorticklabels()[1]
                            tick.set_horizontalalignment('left')
                            divider = make_axes_locatable(ax2)
                            cax = divider.append_axes('right', size='2%', pad=-0.05)
                            cbar = fig.colorbar(mappable=im , cax= cax,orientation="vertical")
                            cbar.set_ticks([-np.pi,0,np.pi])
                            cbar.set_ticklabels([r"$-\pi$","0",r"$\pi$"])
                        else:
                            tick = ax2.xaxis.get_majorticklabels()[-2]
                            tick.set_horizontalalignment('right')
                            tick = ax2.xaxis.get_majorticklabels()[1]
                            tick.set_horizontalalignment('left')

        
                    else:
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='2%', pad=-0.05)
                        cbar = fig.colorbar(mappable=im , cax= cax,orientation="vertical")
                        cbar.set_ticks([-np.pi,0,np.pi])
                        cbar.set_ticklabels([r"$-\pi$","0",r"$\pi$"])

                # Design for the cases of joint x-/y- axis
                if type(ax) !=np.ndarray:
                    if join_y and not separate:
                        if i_ax !=0:
                            ax.set_yticklabels([])
                            ax.set_yticks([])
                            ax.set_ylabel("")
                            ax.ticklabel_format(axis='x', scilimits=[-1, 1])
                
                            tick = ax.xaxis.get_majorticklabels()[-2]
                            tick.set_horizontalalignment('right')
                            tick = ax.xaxis.get_majorticklabels()[1]
                            tick.set_horizontalalignment('left')
                            if i_ax == len(axes)-1:
                                tick = ax.xaxis.get_majorticklabels()[-2]
                                tick.set_horizontalalignment('center')
                        else:
                            ax.ticklabel_format(axis='both', scilimits=[-1, 1])
                            tick = ax.xaxis.get_majorticklabels()[-2]
                            tick.set_horizontalalignment('right')
                    else:
                        ax.ticklabel_format(axis='both', scilimits=[-1, 1])
                    if i_ax == 0 or separate :
                        if type_plot=="fields":
                            ax.set_ylabel("Intensity of fields\ny (m)")
                        elif type_plot == "masks":
                            ax.set_ylabel("Phase mask\ny (m)")
                
                    ax.set_xlabel("x (m)")
                    ax.tick_params(axis='x')
            if save != "":
                if separate:
                    fig.savefig(save + "/"+type_plot+"_plane_"+str(i+1)+".png", transparent=True)
                else:
                    fig.savefig(save + "/"+type_plot+".png", transparent=True)
        if save =="":
            return np.array(figures),np.array(axes)


        

