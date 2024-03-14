import numpy as np
from scipy.special import hermite
import matplotlib.pyplot as plt

class Set_Beams:
    def __init__(self, 
                 id: int,
                 type: str | None,
                 X: list[float] | None,
                 Y: list[float] | None,
                 w: float | None,
                 lam: float | None,
                 num_modes: int | None, 
                 Z: list[float] | np.ndarray | None = None,
                 positions: list[float] | str | None = None,
            
                 **kwargs) -> None:
        # ----------------------------------------
        #
        # Initializing set of beams
        #
        # Inputs:
        #
        # id = Identification of the set of beams
        # type = Type of the set of beams ( Gaussian or Hermite Gaussian at this moment )
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # w = Mode-field diameter of the beams
        # lam = wavelength of the beams
        # num_modes = Number of beam modes 
        # Z = Profile of the aperture where the beam passes in the z direction
        # positions = Defining the positions of the beams
        #
        # ----------------------------------------

        self.type = type
        self.num_modes = num_modes
        self.num_spots = np.sum(np.arange(1,num_modes+1))
        self.positions = positions
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Nx, self.Ny = X.shape
        self.w = w
        self.lam = lam
        self.id = id
        

        
        # Set positions to zero
        if positions is None:
            self.set_pos_zero()
        # If the set of beams will be shaped as a right triangle
        elif positions == "triangle":
            self.set_right_triangle_pos(**kwargs)
        # If positions is different from the number of beams, add positions or number
        elif len(positions) < self.num_spots:
            for i in range(len(positions),self.num_spots):
                self.positions.append([0,0])
        elif len(positions) > self.num_spots:
            self.num_spots = len(positions)

        self.beams = np.empty(self.num_spots,dtype=Beam)

        # Which set of beams to calculate
        if type == "gaussian":
            self.set_Gaussians(self.X, self.Y, self.Z, self.w, self.lam)   

        if type == "hermite":
            self.set_Hermite(self.X, self.Y, self.w, plot = kwargs["plot"])     

    @property 
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        self._id = value


    def set_pos_zero(self) -> None:
        # Reset position of beams to zero
        self.positions= np.zeros((self.num_spots,2), dtype="float32")

    def set_right_triangle_pos(self, x0: float | None = None, y0: float = 0.0, dx:float | None = None, dy: float | None = None, **kwargs) -> None:
        # ----------------------------------------
        #
        # Set the positions of the beams to a right triangle shape
        #
        # Inputs:
        #
        # x0 = x position of the right point of the triangle beam
        # y0 = x position of the right point of the triangle beam
        # dx = distance in x-direction between mode of beams
        # dy = distance in y-direction between mode of beams
        #
        # ----------------------------------------

        # For plotting and positioning proposes
        if "factor" in kwargs:
            factor = kwargs["factor"]
        else:
            factor = 0

        if "factor_mult" in kwargs:
            factor_mult = kwargs["factor_mult"]
        else:
            factor_mult = 1 

        
        if x0 is None:
            x0 = self.X[0,-1] - (self.X[0,-1] - self.X[0,0])/(self.num_modes+factor)
        if dx is None:
            dx = factor_mult * (self.X[0,-1] - self.X[0,0])/(self.num_modes+factor)
        
        if dy is None:
            dy = factor_mult * (self.X[0,-1] - self.X[0,0])/(self.num_modes+factor)

        if num is None:
            num = self.num_spots
        i = 0
        j = 0
        ix = 0
        self.positions = []
        while i < num:
            iy = -1 * ix + 2*j

            x = -ix*dx + x0
            y = -iy*dy/2 + y0

            if j < ix:
                j+=1
            else:
                j=0
                ix+=1
                
            self.positions.append([x,y])


            i+=1
        self.positions = np.array(self.positions)

    def set_Gaussians(self, X: list[float] | np.ndarray, Y: list[float] | np.ndarray, Z: list[float] | np.ndarray, w: float, lam: float) -> None:

        # ----------------------------------------
        #
        # Initializing set of Gaussian beams
        #
        # Inputs:
        #
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # Z = Profile of the aperture where the beam passes in the z direction
        # w = Mode-field diameter of the beams
        # lam = wavelength of the beams
        #
        # ----------------------------------------

        for i in range(self.num_spots):
            pos_x = self.positions[i][0]
            pos_y = self.positions[i][1]
            XG = X - pos_x
            YG = Y - pos_y
            self.beams[i] = Beam.GaussianBeam(Z, XG, YG, w, lam)

    def set_Hermite(self, X: list[float] | np.ndarray, Y: list[float] | np.ndarray, w : float, plot: bool = False ) -> None:

        # ----------------------------------------
        #
        # Initializing set of Gaussian beams
        #
        # Inputs:
        #
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # w = Mode-field diameter of the beams
        #
        # ----------------------------------------
        idx = 0


        for i in range(self.num_modes):
            mgIDX = i
            for j in range(i+1):
                m = mgIDX - j
                n = mgIDX - m

                pos_x = self.positions[idx][0]
                pos_y = self.positions[idx][1]

                XG = X - pos_x
                YG = Y - pos_y
                self.beams[idx] = Beam.HermiteGaussian(n,m,XG,YG,w)
                idx+=1


    def get_total(self) -> np.ndarray:
        # Calculate the total ensemble of the beams
        Total = np.zeros((self.Nx,self.Ny),dtype="float32")
        for i in range(self.num_spots):
            Total += np.abs(self.beams[i].result)**2
        return Total

    def plot(self, X = None, Y = None, perMode: bool = False):
        # Return the figure and axis of the total ensemble of the beams
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if not perMode:
            Total = self.get_total()
            fig,ax = plt.subplots()
            ax.pcolormesh(X,Y,Total)
            return fig,ax
        else:
            pass


class Beam:
    def __init__(self, X: list[float] | np.ndarray, Y: list[float] | np.ndarray, result: list[float] | np.ndarray) -> None:
        # ----------------------------------------
        #
        # Generation of a Gaussian Beam
        #
        # Inputs:
        #
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # result = Profile of the beam at a certain z distance
        #
        # ----------------------------------------
        self.X = X
        self.Y = Y
        self.result = result
    
    
    @classmethod
    def GaussianBeam(self,Z,X,Y, w0i, lam):
        # ----------------------------------------
        #
        # Generation of a Gaussian Beam
        #
        # Inputs:
        #
        # Z = Profile of the aperture where the beam passes in the z direction
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # w0 = Beam radius when amplitude has dropped 1/e of its value on the optical axis
        # lam = wavelength of the beam
        #
        # ----------------------------------------

        result = np.zeros(X.shape,dtype="csingle")

        w0=w0i/2.0
        # Wave vector obtained from the wavelength
        k = 2 * np.pi / lam

        # Rayleigh length
        zr = np.pi * w0**2 / lam

        # Beam radius dependent of the position
        wz = w0 * np.sqrt( 1 + (Z/zr)**2)

        # Radius of curvature of phase front
        Rz = Z * ( 1 + ( zr/Z)**2)

        # Radius grid of the beam perpendicular to the direction of the beam
        R2 = X**2 + Y**2
        # Transverse phase
        PhiT = k * R2 / ( 2 * Rz)

        # Longitudinal phase
        PhiL = k * Z - np.arctan(Z/zr)
                
        result[:,:] = w0/wz * np.exp( -1.0 * R2/ wz**2 ) * np.exp( 1j *( - PhiL - PhiT ))

        
        norm = np.sqrt( np.sum( np.abs(result)**2))
        if norm !=0:
            result = result/norm

        return self(X,Y,result)
    
    @classmethod
    def HermiteGaussian(self, m, n, X, Y, w1i):
        # ----------------------------------------
        #
        # Generation of an Hermite Gaussian Beam
        #
        # Inputs:
        #
        # m,n = modes of the Hermite Gaussian beam
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # w1i = Beam radius when amplitude has dropped 1/e of its value on the optical axis
        #
        # ----------------------------------------
        w1=w1i/2.0
        result = np.zeros(X.shape,dtype="csingle")

        # Hermite functions
        Hn = hermite(n,monic = False)
        eval_Hn = np.polyval(Hn, np.sqrt(2)*X/w1)

        Hm = hermite(m, monic=False)
        eval_Hm = np.polyval(Hm, np.sqrt(2)*Y/w1)

        # Gaussian function
        Gaussian = np.exp(- ((np.sqrt(2)*X/w1)**2 + (np.sqrt(2)*Y/w1)**2)/2)
        
        result[:,:] = eval_Hn * eval_Hm * Gaussian

        norm = np.sqrt(np.sum(np.abs(result)**2))

        if norm !=0:
            result = result/norm 
        
        return self(X,Y,result)
    
    @classmethod
    def freeSpace(self,Nx, Ny, X, Y, dz, lam):
        # ----------------------------------------
        #
        # Transfer function when beam propagates in vacuum
        #
        # Inputs:
        #
        # Nx, Ny = Number of elements in X and Y grids
        # X, Y = Grid perpendicular to the direction of the beam (z)
        # dz = Distance between masks that the beams have traveled
        # lam = wavelength of the beam
        #
        # ----------------------------------------
        result = np.zeros((Nx,Ny),dtype="csingle")
        fs = Nx / ( np.max(X) - np.min(X)   )
        vx = fs * ( np.arange(-Nx/2,Nx/2,1)) / Nx
        fs = Ny / ( np.max(Y) - np.min(Y)   )
        vy = fs * ( np.arange(-Ny/2,Ny/2,1)) / Ny

        Vx, Vy = np.meshgrid(vx,vy)
 
        arg = 1.0/lam**2 - Vx**2 - Vy**2
        result = np.exp( -1j * 2 * np.pi * np.sqrt(arg) * dz)
        
        return self(Vx, Vy, result)