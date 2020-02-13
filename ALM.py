#from totalvariation import *
import numpy as np
from scipy import sparse, linalg, signal, misc, ndimage, fftpack
import matplotlib.pyplot as plt
import time
from functools import partial
from utils import *

def A(f):#kernel):
    #return signal.fftconvolve(f,kernel,mode='same')
    #out1 = np.fft.rfft2(f, s=shape)
    out1 = fftpack.fftn(f, shape=shape)
    return fftpack.ifftn(out1*fker)[s2[0]//2:(shape[0]+1-s2[0]//2),s2[1]//2:(shape[1]+1-s2[1]//2)]

def Astar(f):#,kernel):
    #return signal.fftconvolve(f,kernel[::-1],mode='same')
    #out1 = np.fft.rfft2(f, s=shape)
    out1 = fftpack.fftn(f, shape=shape)
    return fftpack.ifftn(out1*fker_conj)[s2[0]//2:(shape[0]+1-s2[0]//2),s2[1]//2:(shape[1]+1-s2[1]//2)]

def AstarA(f):#,kernel):
    return Astar(A(f))
    #DOES NOT WORK
    #out1 = np.fft.rfft2(f, s=shape)
    out1 = fftpack.fftn(f, shape=shape)
    return fftpack.ifftn(out1*fker2)[s2[0]//2:(shape[0]+1-s2[0]//2),s2[1]//2:(shape[1]+1-s2[1]//2)]

def AstarA1d(f):
    return AstarA(f.reshape(nx,ny))

def operator(z, alpha):
    return AstarA1d(z).reshape(nx*ny) + np.multiply(alpha,z)

def getz(f, p, u,alpha):
    return 0

def getu(rhs,lam):
    return ptv.tv1_2d(rhs,lam)

def cg(z,rhs,alpha):
    TOL = 0.01
    r = -operator(z,alpha) + rhs 
    r0 = np.linalg.norm(r,'fro')**2
    p = r
    rr_next = np.linalg.norm(r,'fro')**2
    for i in range(nx*ny):
        Ap = operator(p,alpha)
        rr_curr = rr_next
        beta = rr_curr/np.dot(p.reshape(nx*ny),Ap.reshape(nx*ny))
        z = np.add(z, beta*p)
        r = np.add(r, -beta*Ap)
        rr_next = np.linalg.norm(r,'fro')**2
        if rr_next/r0 < TOL:
            break
        gamma = rr_next/rr_curr
        p = r + gamma*p
    return z

def augmentedLagrangian(f,lam):
    alpha = 1.0
    tol = 0.1
    maxiter = 3
    z,z_next = np.zeros((nx,ny)), np.empty((nx,ny))
    u,u_next = np.copy(f), np.empty((nx,ny))
    p,p_next = np.zeros((nx,ny)),np.empty((nx,ny))
    rhs = np.empty((nx,ny)) 
    for i in range(maxiter):
        operator_k = sparse.linalg.LinearOperator((nx*ny,nx*ny),partial(operator,alpha=alpha))
        rhs = (Astar(f) - p + alpha * u).reshape(nx*ny)
        z_next,info = sparse.linalg.gmres(operator_k,rhs, x0 = z.reshape(nx*ny),tol=tol)
        print(info)
        z_next = z_next.reshape(nx,ny)
        rhs =  alpha*z_next + p  
        u_next = getu(rhs, lam)/alpha
        p_next = p - alpha*(u_next - z_next)
        print(np.linalg.norm(u_next-u))
        print(np.linalg.norm(z_next-z))
        print(np.linalg.norm(p_next-p))
        alpha = alpha*1.3
        tol = tol*0.3
        u = u_next
        z = z_next
        p = p_next
    return z_next
if __name__ == "__main__":
    #Originalf = misc.face(gray=True)
    Originalf = misc.imread('images/einstein.jpg', 'jpg')
    Originalf = Originalf/255.0
    f = np.copy(Originalf)
    nx = f.shape[0]
    ny = f.shape[1]
    hx = 1./(nx+1)
    hy = 1./(ny+1)
    dim = 2
    sigma = 1.0
    kx = 100
    ky = 100
    ker = np.outer(signal.gaussian(kx, 5*sigma), signal.gaussian(ky, sigma))
    ker = ker/np.linalg.norm(ker)

    s1 = np.array(f.shape)
    s2 = np.array(ker.shape)
    shape = s1 + s2 - 1
    print(shape)
    #print(np.linalg.norm(f))
    #print(np.linalg.norm(ker))
    fker = fftpack.fftn(ker,shape=shape)
    #plt.imshow(np.abs(fftpack.ifftn(fker)))
    #plt.show()
    fker_conj = fftpack.fftn(ker[::-1,::-1],shape=shape)
    fker2 = np.abs(fker)**2#np.fft.rfft2(ker*ker,s=shape)
    f = A(f)
    plt.imshow(np.abs(f), cmap='gray')
    plt.show()

    sigma = 0.01
    f = add_gaussian_noise(f, sigma)
    result = augmentedLagrangian(f,0.5)
    print(np.linalg.norm(Originalf-f))
    print(np.linalg.norm(Originalf-result))
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].imshow(Originalf, cmap = 'gray')
    #ax[0].set_title("Original")
    ax[1].imshow(np.abs(f), cmap ='gray')
    #ax[1].set_title("Noisy, sigma = " + str(sigma))
    ax[2].imshow(np.abs(result), cmap = 'gray')
    #ax[2].set_title("Solution, lambda = " + str(lam)[0:5])
    plt.show()

