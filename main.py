import numpy as np
from PIL import Image
import imageio
from utils import *
from parameterselection import optTV
from solvers import *
from utils import D_zero_boundary,Dast_zero_boundary, D_convolution, Dast_convolution
#from quadratic import f2py_quadratic_denoise


if __name__ == '__main__':  
    np.random.seed(315)
    u = imageio.imread('images/lenna.png')/255.0
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    test = np.zeros((6,6))
    test[0,0] = 1.0; test[0, -1] = 1.0; test[-1,0] = 1.0; test[-1,-1] = 1.0
    lam = 100.0
    t = time.time()
    u_rec = py_quadratic_cg_denoise(f, 1.0,1.0, precond = True,tol = 1.0e-8, maxiter = 1000, verbose = True)
    
    print(time.time() - t, "preconditioned") 
        
    
