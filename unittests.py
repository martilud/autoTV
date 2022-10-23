from utils import grad as GRAD
from utils import div as DIV
from utils import D, Dast
import imageio
#from chambollepock import *
#from quadraticTV import *
import numpy as np
import time
from PIL import Image
import os
import numpy.random
#from parameterselection import *
#import scipy.ndimage
#from skimage.metrics import structural_similarity as ssim
#from interpolation import *
#from skimage.transform import resize
#from numpy.fft import fft2, ifft2, fftshift, rfftn, irfftn
#from scipy.fftpack import dct, idct
#from scipy.signal import fftconvolve, convolve, gaussian
#from solve import grad, div

def run_unittests():
    testD()

def test_f2py():
    TOL = 1e-5
    u = np.array(imageio.imread('images/lenna.png')/255.0,dtype=np.float32, order='F')
    n = u.shape[0]
    du = np.zeros((2,n,n), dtype=np.float32, order = 'F')
    res = np.zeros((n,n), dtype=np.float32, order = 'F')
    grad(u,du,n) 
    assert(np.linalg.norm(du - GRAD(u)) < TOL)
    div(du,res,n)
    assert(np.linalg.norm(res - DIV(du)) < TOL)
    t = time.time()
    fres = f2py_cp_denoise(u, 0.1)
    print(time.time() - t)
    t = time.time()
    pyres = ChambollePock_denoise(u, 0.1)
    print(time.time() - t)
    assert(np.linalg.norm(fres-pyres)/n**2 < TOL)

def testD():
    u = np.array(imageio.imread('images/lenna.png')/255.0,dtype=np.float32, order='F')
    shape = u.shape
    p = numpy.random.normal(0,1,2*np.prod(shape)).reshape((2,shape[0],shape[1]))
    Du = D(u)
    inner = np.sum(np.multiply(Du[0],p[0])) + np.sum(np.multiply(Du[1],p[1]))
    print(inner)
    Dstarp = Dast(p)
    innerstar = np.sum(np.multiply(u,Dstarp))
    print(innerstar)
run_unittests()
