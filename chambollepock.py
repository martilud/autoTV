from utils import *
from numpy.fft import rfftn, irfftn
#from solve import chambollepock_denoise

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

def py_cp_denoise(f, t, u0 = None, tau = 0.25, sig = 0.25, theta = 1.0, tol = 1.0e-10):

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
    n = f.shape[0] 
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 1000
    convlist = []
    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = p_hat / np.maximum(1,norm1(p_hat))
        divp = div(p)
        u = 1.0/(1.0 + t*tau) * (u + tau * divp + tau * t * f) 
        u_hat = u + theta*(u - u_prev)
        convlist.append(np.linalg.norm(-divp + t*(u-f)))
        if (convlist[i]/convlist[0]<tol):
            break
    return u

def py_cp_denoise_dp(f, t, u_true,noise_sig = 0.0, dp_tau = 1.0, u0 = None, tau = 0.25, sig = 0.25, theta = 1.0, tol = 1.0e-10):

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
    n = f.shape[0] 
    cc = dp_tau * n* n * noise_sig**2
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 1000
    maxiter_newton = 50
    t_hat = t
    list = []
    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p =p_hat / np.maximum(1, norm1(p_hat)) 
        divp = div(p)
         
        # Find t by Newton iteration on discrepancy principle
        #if (np.linalg.norm(u - f, 'fro')**2 < cc):
        #    t = 0.0
        #else:
        for j in range(maxiter_newton):
            t_prev = t
            temp = np.linalg.norm(u + tau* divp -f,'fro')**2
            e = 1/(1 + t * tau)**2 * temp - cc
            de = - 2*tau/(1 + t * tau)**3 * temp
            t = t_prev - e/de
            if (t < 0.0):
                alpha = 1.0
                for m in range(50):
                    t = t_prev - alpha * e/de
                    if 0.0 < t:
                        break
                    alpha*=0.5
            if (np.abs(e) < 1e-12):
                break

        u = 1.0/(1.0 + t*tau) * (u + tau * divp + tau * t * f)
        #u_hat = 2*u - u_prev
        list.append(np.linalg.norm(-divp + t*(u-f),'fro')**2)
        if list[i]/list[0] < 1e-12:
            break
    #plt.semilogy(list)
    #plt.show()
    return u,t


def py_pg_denoise_dp(f, t, u_true, noise_sig = 0.0, dp_tau = 1.0, u0 = None, tau = 0.25, tol = 1.0e-10):

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
    n = f.shape[0] 
    cc = dp_tau * n* n * noise_sig**2
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = div(p)
    maxiter = 1000
    maxiter_newton = 10
    conv_u = []
    conv_t = []
    u_check = []
    duality_list=[]
    iters = 0
    for i in range(maxiter):
        t_bigprev = np.copy(t)
        u_prev = np.copy(u)
        p_hat = p + tau*grad(u)
        p = p_hat / np.maximum(1, norm1(p_hat)) 
        divp = div(p)
        #t_list = np.linspace(-100,100,1000)
        #e_list = np.zeros(1000)
        #for m in range(1000):
        #        e_list[m] = (1/t_list[m]**2)*np.linalg.norm(divp,'fro')**2 - cc
        #plt.plot(t_list,e_list)
        #plt.show()
        normdivp =  np.linalg.norm(divp,'fro')**2
        for j in range(maxiter_newton):
            t_prev = t
            k = (1.0/(t*t))*normdivp - cc
            dk = -(2.0/(t*t*t)) * normdivp
            t = t_prev - k/dk
            alpha = 1.0
            if (t < 0.0):
                alpha = 1.0
                for m in range(20):
                    t = t_prev - alpha * k/dk
                    if t > 0.0:
                        break
                    alpha*=0.5
            if (np.abs(k) < 1e-12):
                break
        u = f + (1/t)*divp 
        #p_hat = p + tau*grad(u)
        #p = p_hat / np.maximum(1, norm1(p_hat))
        #divp = div(p)
        #duality_list.append(dualitygap_denoise(t,u,divp,f))
        duality_list.append(np.linalg.norm(-divp + t*(u-f),'fro')**2)
        iters +=1
        #conv_u.append(np.linalg.norm(u - u_prev,'fro')**2/np.linalg.norm(u_prev,'fro')**2)
        #conv_t.append(np.abs(t - t_bigprev)**2/np.abs(t_bigprev)**2)
        #u_check.append(np.linalg.norm(u - u_true,'fro')**2/np.linalg.norm(u_true,'fro')**2)
        #if conv_u[i] < 1e-12:
        #    break
        if duality_list[i]/duality_list[0] < 1e-12:
            break
    #plt.semilogy(duality_list)
    #plt.show()
    #plt.semilogy(u_check)
    #plt.show()
    #plt.semilogy(conv_t)
    #plt.show()
    return u,t


def f2py_cp_denoise(f, lam, u0 = None, tau = 0.25, sig = 0.25, theta = 1.0, acc = False, tol = 1.0e-10):
    n = f.shape[0]
    if u0 is None:
        res = np.zeros(f.shape, dtype = np.float32, order = 'F')
    else:
        res = u0
    chambollepock_denoise(f,res, lam, tau, sig, theta, acc, tol, n)
    return res

    
