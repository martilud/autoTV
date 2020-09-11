from utils import *
from numpy.fft import rfftn, irfftn
from speed import chambollepock_denoise

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

def ChambollePock_convolution_edge(f, lam, fker, fker_star,shape, u0 = None,  tau = 0.25, sig = 0.25, theta = 1.0, acc = False, tol = 1.0e-10):
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

    fk_star_times_ff = np.multiply(fker_star, rfftn(edge_pad_and_shift(f,m),shape))     
    one_plus_tau_fker_squared = np.ones(fker.shape) + tau * np.abs(fker)**2 

    if acc:
        gam = 0.5

    maxiter = 100
    for i in range(maxiter):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = lam * p_hat / np.maximum(lam,norm1(p_hat))
        divp = div(p)
        u = center(irfftn(np.divide(rfftn(np.pad(u,((m-1)//2,(m-1)//2), 'edge') + np.pad(tau * divp,((m-1)//2,(m-1)//2), 'edge'), shape) + tau * fk_star_times_ff, one_plus_tau_fker_squared)),n,m)
        if(np.linalg.norm(u - u_prev, 'fro') < tol):
            break
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
    return center(u,n,m)   

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

def ChambollePock_denoise(f, lam, tau = 0.50, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-4, u0 = None):

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

def f2py_cp_denoise(f, lam, u0 = None, tau = 0.30, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-10):
    n = f.shape[0]
    if u0 is None:
        res = np.zeros(f.shape, dtype = np.float32, order = 'F')
    else:
        res = u0
    chambollepock_denoise(f,res, lam, tau, sig, theta, acc, tol, n)
    return res

    
