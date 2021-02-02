import numpy as np
from utils import div, grad, center, edge_pad_and_shift
from numpy.fft import rfftn, irfftn
import matplotlib.pyplot as plt
from speed import quadreg_denoise
def quadfunc_denoise(f, lam):
    return -lam * div(grad(f)) + f

def quadfunc_convolution(f, lam, fker, fker_star):
    n = f.shape[0]
    m = fker.shape[0] - n + 1
    shape = (n + m -1, n + m -1)
    return -lam *div(grad(f)) + irfftn(rfftn(edge_pad_and_shift(f,m), shape)*np.abs(fker)**2)[:n,:n]

def quadfunc_alt(f, lam):
    result = np.zeros(f.shape)
    n = f.shape[0]
    # FOR ALL
    result[:,:] = f[:,:] + lam*4*f[:,:]
    #CORNERS
    result[0,0]-= lam * (2*f[0,1] + 2*f[1,0])
    result[0,-1]-= lam * (2*f[0,-2] + 2*f[1,-1])
    result[-1,0]-= lam * (2*f[-2,0] + 2*f[-1,1])
    result[-1,-1]-= lam * (2*f[-1,-2] + 2*f[-1,-2])
    #BOUNDARIES
    result[0, 1:-1] -= lam *(2*f[1,1:-1] + f[0,:-2] + f[0,2:])
    result[-1, 1:-1] -= lam *(2*f[-2,1:-1] + f[-1,:-2] + f[-1,2:])
    result[1:-1, 0] -= lam *(2*f[1:-1,1] + f[:-2,0] + f[2:,0])
    result[1:-1, -1] -= lam *(2*f[1:-1,-2] + f[:-2,-1] + f[2:,-1])
    #INSIDE
    result[1:-1,1:-1] -= lam * (f[:-2,1:-1] + f[2:, 1:-1] + f[1:-1,:-2] + f[1:-1,2:]) 
    return result

def quadraticRegularizer_denoise(f, lam, tol = 1.0e-4):
    u = np.zeros(f.shape)
    Au = quadfunc_denoise(u,lam)
    r = f - Au
    p = r
    rr_next = np.linalg.norm(r,'fro')**2
    r0 = rr_next
    MAXITER = 100
    for i in range(MAXITER):
        Ap = quadfunc_denoise(p,lam)
        rr_curr = rr_next
        alpha = rr_curr/np.sum(np.multiply(p,Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        rr_next = np.linalg.norm(r,'fro')**2
        if rr_next/r0 < tol:
            break
        beta = rr_next/rr_curr
        p = r + beta*p
    return u

def quadraticRegularizer_convolution(f, lam, fker, fker_star, shape, tol = 1.0e-10):
    u = np.zeros(f.shape)
    n = f.shape[0]
    m = fker.shape[0] - n + 1
    shape = (n + m - 1, n + m - 1)

    Au = quadfunc_convolution(u,lam, fker, fker_star)
    rhs = center(irfftn(rfftn(edge_pad_and_shift(f,m),shape) * fker_star),n,m)
    r = rhs - Au
    r0 = np.linalg.norm(f - Au,'fro') 
    p = r
    rr_next = np.linalg.norm(r,'fro')**2
    MAXITER = 100
    for i in range(MAXITER):
        Ap = quadfunc_convolution(p,lam, fker, fker_star)
        rr_curr = rr_next
        alpha = rr_curr/np.sum(np.multiply(p,Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r,'fro')/r0 < tol:
            break
        rr_next = np.linalg.norm(r,'fro')**2
        beta = rr_next/rr_curr
        p = r + beta*p
    return u

def counterexample(f,lam,tol = 1.0e-4):
    u = np.zeros(f.shape)
    Au = quadfunc_denoise(quadfunc_denoise(u,lam), lam)
    r = f - Au
    r0 = np.linalg.norm(f - Au,'fro') 
    p = r
    rr_next = np.linalg.norm(r,'fro')**2
    MAXITER = 100
    for i in range(MAXITER):
        Ap = quadfunc_denoise(quadfunc_denoise(p,lam), lam)
        rr_curr = rr_next
        alpha = rr_curr/np.sum(np.multiply(p,Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r,'fro')/r0 < tol:
            break
        rr_next = np.linalg.norm(r,'fro')**2
        beta = rr_next/rr_curr
        p = r + beta*p
    return -div(grad(u))

def f2py_quadratic_denoise(f, lam, u0 = None, tol = 1.0e-10):
    n = f.shape[0]
    if u0 is None:
        res = np.zeros(f.shape, dtype = np.float32, order = 'F')
    else:
        res = u0
    quadreg_denoise(f, res, lam, tol, n)
    return res

