from utils import *
from projectedGradient import *
from numba import jit
from numpy.fft import rfftn, irfftn

def center(f,n,m):
    """
    Returns centered image of size nxn that has
    been padded to size n+m-1xn+m-1
    """
    return f[m//2:m//2+n, m//2:m//2+n]

def A(ff,fker,n,m):
    shape = (n+m-1,n+m-1)
    return center(fftpack.irfftn(ff*fker,shape))


def A_star(ff,fker_star,n,m):
    shape = (n+m-1,n+m-1)
    return center(fftpack.irfftn(ff*fker_star,shape))

def A_square(ff,fker_square):
    return 0

def ChambollePock_convolution(f, lam, fker, fker_star, shape, tau = 0.30, sig = 0.20, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):
    if u0 is  None:
        u = np.zeros(f.shape, f.dtype)
        u_prev = np.zeros(f.shape, f.dtype) 
        u_hat = np.zeros(f.shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    
    n = f.shape[0]
    m = shape[0] - n + 1

    fk_star_times_ff = np.multiply(fker_star, rfftn(f,shape)) # This should be fixed so the edges are done better, see main function
    one_plus_tau_fker_squared = np.ones(fker.shape) + tau * np.abs(fker)**2 # I think this line is funky, i think the scaling of the ones may be off
    #one_plus_tau_fker_squared = rfftn(temp,shape) + tau * np.abs(fker)**2 

    if acc:
        gam = 0.5

    maxiter = 100
    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam,norm1(p_hat))
        divp = div(p)
        
        u = center(irfftn(np.divide(rfftn(np.pad(u + tau * divp,((m-1)//2,(m-1)//2)), shape) + tau * fk_star_times_ff, one_plus_tau_fker_squared)),n,m)
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
    return u   

def edge_pad_and_shift(f, m):
    """
    pads image to size n+m-1xn+m-1 with edge padding
    """
    f_pad = np.pad(f,((m-1)//2,(m-1)//2), 'edge')
    f_pad = np.roll(f_pad, -(m-1)//2, axis = 0)
    f_pad = np.roll(f_pad, -(m-1)//2, axis = 1)
    return f_pad

def ChambollePock_convolution_edge(f, lam, fker, fker_star, shape, tau = 0.3, sig = 0.25, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):
    if u0 is  None:
        u = np.zeros(f.shape, f.dtype)
        u_prev = np.zeros(f.shape, f.dtype) 
        u_hat = np.zeros(f.shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    
    n = f.shape[0]
    m = shape[0] - n + 1

    fk_star_times_ff = np.multiply(fker_star, rfftn(edge_pad_and_shift(f,m),shape)) # This should be fixed so the edges are done better, see main function
    one_plus_tau_fker_squared = np.ones(fker.shape) + tau * np.abs(fker)**2 # I think this line is funky, i think the scaling of the ones may be off
    #one_plus_tau_fker_squared = rfftn(temp,shape) + tau * np.abs(fker)**2 

    if acc:
        gam = 0.5

    maxiter = 100
    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam,norm1(p_hat))
        divp = div(p)
        u = center(irfftn(np.divide(rfftn(np.pad(u,((m-1)//2,(m-1)//2), 'edge') + np.pad(tau * divp,((m-1)//2,(m-1)//2), 'edge'), shape) + tau * fk_star_times_ff, one_plus_tau_fker_squared)),n,m)
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
    return u   

def ChambollePock_convolution_alt(f, lam, fker, fker_star, shape, tau = 0.30, sig = 0.20, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):
    if u0 is  None:
        u = np.zeros(shape, f.dtype)
        u_prev = np.zeros(shape, f.dtype) 
        u_hat = np.zeros(shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)
    p = np.zeros((2,) + shape, f.dtype)
    p_hat = np.zeros((2,) + shape, f.dtype)
    divp = div(p)
    
    n = f.shape[0]
    m = shape[0] - n + 1
    
    fk_star_times_ff = np.multiply(fker_star, rfftn(f,shape)) # This should be fixed so the edges are done better, see main function
    one_plus_tau_fker_squared = np.ones(fker.shape) + tau * np.abs(fker)**2 # I think this line is funky, i think the scaling of the ones may be off
    #one_plus_tau_fker_squared = rfftn(temp,shape) + tau * np.abs(fker)**2 
    if acc:
        gam = 0.5

    maxiter = 100
    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam,norm1(p_hat))
        divp = div(p)
        
        u = irfftn(np.divide(rfftn(u + tau * divp, shape) + tau * fk_star_times_ff, one_plus_tau_fker_squared))
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
    return u   

def ChambollePock_denoise_conv(f, lam, tau = 0.50, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):
    if u0 is  None:
        #u = np.copy(f)
        #u_prev = np.copy(f)
        #u_hat = np.copy(f)
        u = np.zeros(f.shape, f.dtype)
        u_prev = np.zeros(f.shape, f.dtype) 
        u_hat = np.zeros(f.shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)

    #p_hat = sig*grad(u_hat)
    #p = lam * p_hat / np.maximum(lam, better_norm1(p_hat))
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 500
    dualitygap_list= np.zeros(maxiter+1)
    dualitygap_list[0] = dualitygap_denoise(lam,u,divp,f)
    psnr_list = np.zeros(maxiter)
    #psnr_list[0] = psnr(u,f)
    if acc:
        gam = 0.5
    i = 0
    res= tol + 1.0
    plot_iterations = [10, 100, 500]
    while (i < maxiter and res > tol): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam, norm1(p_hat))
        divp = div(p)
        u = 1/(1 + tau) * (u + tau * divp + tau *f) 
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
        dualitygap_list[i+1] = dualitygap_denoise(lam,u,divp,f)
        res = dualitygap_list[i+1]/dualitygap_list[0]
        if (i+1 in plot_iterations):
            plt.imsave("Lenna" + str(i) + ".png", u, cmap = "gray")
        psnr_list[i] = psnr(u,u_prev)
        i+=1
        #print(i)
    return u, dualitygap_list, psnr_list, i 

def ChambollePock_denoise(f, lam, tau = 0.50, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-1, u0 = None):

    if u0 is  None:
        #u = np.copy(f)
        #u_prev = np.copy(f)
        #u_hat = np.copy(f)
        u = np.zeros(f.shape, f.dtype)
        u_prev = np.zeros(f.shape, f.dtype) 
        u_hat = np.zeros(f.shape, f.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_hat = np.copy(u0)
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 100
    dualitygap_init = dualitygap_denoise(lam,u,divp,f)
    if acc:
        gam = 0.5
    res= tol + 1.0
    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam,norm1(p_hat))
        divp = div(p)
        u = 1.0/(1.0 + tau) * (u + tau * divp + tau *f) 
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
        if (i%10 == 0):
            res = dualitygap_denoise(lam,u,divp,f)/dualitygap_init
            if (res < tol):
                break
    return u   

if __name__ == "__main__":
    ker = np.outer(signal.gaussian(2*nx,sigma), signal.gaussian(2*ny, sigma))
    ker = ker/np.linalg.normsum(ker)


