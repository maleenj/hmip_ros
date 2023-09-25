import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VectorField3D:

    def __init__(self, n_x, n_y, n_z,disc_param):
       
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        # Generate the 3D grid
        x = np.linspace(-n_x, n_x,np.int32((2*n_x)/disc_param)+1)
        y = np.linspace(-n_y, n_y,np.int32((2*n_y)/disc_param)+1)
        z = np.linspace(-n_z, n_z,np.int32((2*n_z)/disc_param)+1)

        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        #self.u, self.v, self.w = np.meshgrid(x, y, z, indexing='ij')
        self.u=self.X*0
        self.v=self.Y*0
        self.w=self.Z*0

    def calc_distances(self, wp_x, wp_y, wp_z):

        WP_X=np.ones_like(self.X)*wp_x
        WP_Y=np.ones_like(self.Y)*wp_y
        WP_Z=np.ones_like(self.Z)*wp_z

        self.distancemesh=np.sqrt((self.X-WP_X)**2+(self.Y-WP_Y)**2+(self.Z-WP_Z)**2)

        return self.distancemesh
    
    def calc_alpha_sigmoid(self, distancemesh, beta):

        alpha = 1/(1+np.exp(distancemesh/beta))
        
        return alpha
    
    
    def updateDBVF(self, v_x, v_y, v_z, wp_x, wp_y, wp_z, alpha):

        self.u = np.multiply(alpha,v_x)+ np.multiply((1-alpha),self.u)
        self.v = np.multiply(alpha,v_y)+ np.multiply((1-alpha),self.v)
        self.w = np.multiply(alpha,v_z)+ np.multiply((1-alpha),self.w)
            
        return self.u, self.v, self.w
    
    def updateVF(self, v_x, v_y, v_z, wp_x, wp_y, wp_z, alpha):

        distances = v_x*(self.X-wp_x) + v_y*(self.Y-wp_y) + v_z*(self.Z-wp_z)
        behind_plane= distances < 0

        self.u[behind_plane] = np.multiply(alpha[behind_plane],v_x)+ np.multiply((1-alpha[behind_plane]),self.u[behind_plane])
        self.v[behind_plane] = np.multiply(alpha[behind_plane],v_y)+ np.multiply((1-alpha[behind_plane]),self.v[behind_plane])
        self.w[behind_plane] = np.multiply(alpha[behind_plane],v_z)+ np.multiply((1-alpha[behind_plane]),self.w[behind_plane])
            
        return self.u, self.v, self.w, behind_plane