import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
import scipy.interpolate
from utils import *
from numpy.fft import fftn

def discrepancy_rule(f,method, noise_std = 0.0, param = 1.0, lam_init= 1.0, q = 0.90, plot = False, **method_kwargs):
    """
    Implementation of the discrepancy principle
    noise_std is the estimate (or the true value) of the noise parameter
    """
    nn = np.prod(f.shape)
    i = 0
    lam = lam_init
    max_iter = 100
    error_level = param * noise_std *np.sqrt(nn)
    disp_list = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    for i in range(1, max_iter):
        print(i, lam)
        u = method(f, lam, **method_kwargs)
        lam = lam_init * (q ** i)
        if plot:
            t_list[i] = 1.0 / (1.0 + lam_init * (q ** i))
            disp_list[i] = np.linalg.norm(u-f)
        else:
            if np.linalg.norm(u - f) < error_level:
                break

    if plot:
        plt.plot(t_list[1:], disp_list[1:], label = r'$|S(t)v - v|$')
        plt.hlines(error_level, min(t_list[1:]), max(t_list[1:]), label= r'$\tau n \sigma$' )
        plt.legend()
        plt.xlabel(r'$t$')
        plt.grid()
        plt.savefig('disp.png')
        plt.show()
    t = 1.0 / (1.0 + lam_init * (q ** i))
    return u, t

def discrepancy_rule_convolution(f,method, fker, fker_star, shape, noise_std = 0.0, param = 1.0, lam_init= 0.01, q = 0.95, plot = False, **method_kwargs):
    """
    Implementation of the discrepancy principle
    noise_std is the estimate (or the true value) of the noise parameter
    """
    n = f.shape[0]
    m = shape[0] - n + 1
    i = 0
    lam = lam_init
    max_iter = 10
    error_level = param * noise_std *n
    disp_list = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    for i in range(1, max_iter):
        u = method(f, lam, fker, fker_star, shape,**method_kwargs)
        lam = lam_init * (q ** i)
        if plot:
            t_list[i] = 1.0 / (1.0 + lam_init * (q ** i))
            print(t_list[i])
            Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
            disp_list[i] = np.linalg.norm(Au-f)
        #else:
        #    if np.linalg.norm(u - f) < error_level:
        #        break

    if plot:
        plt.plot(t_list[1:], disp_list[1:], label = r'$|S(t)v - v|$')
        plt.hlines(error_level, min(t_list[1:]), max(t_list[1:]), label= r'$\tau n \sigma$' )
        plt.legend()
        plt.xlabel(r'$t$')
        plt.grid()
        plt.savefig('disp.png')
        plt.show()
    t = 1.0 / (1.0 + lam_init * (q ** i))
    return u, t

def monotone_rule(f, method, noise_std = 0.0, param = 1.0, lam_init= 2.0, q = 0.95, **method_kwargs):
    """
    Implementation of the discrepancy principle
    noise_std is the estimate (or the true value) of the noise parameter
    """
    nn = np.prod(f.shape)
    lam = lam_init
    max_iter = 50
    error_level = param * noise_std * np.sqrt(nn)
    me_tol = 1e-5
    u_old = method(f, lam, **method_kwargs)
    i = 1
    list = np.zeros(max_iter)
    while (i < max_iter):
        print(i)
        lam = lam_init * (q ** i)
        u_new = method(f, lam, **method_kwargs)
        list[i] = np.sum(np.multiply(u_old - f, u_old - u_new))/np.linalg.norm(u_old - u_new, 'fro')
        if np.sum(np.multiply(u_old - f, u_old - u_new))/np.linalg.norm(u_old - u_new, 'fro') < error_level:

            break
        if (lam <= me_tol):
    	    break
        i += 1
        u_old = u_new
    plt.plot(list)
    plt.show()
    t = 1.0 / (1.0 + lam_init * (q ** i))
    return u_new, t

def quasi_optimality(f, method,lam_init = 1.0, q = 0.90, plot = False, **method_kwargs):
    """
    Implementation of quasi optimality for parameter selection
    """
    
    lam = lam_init
    max_iter = 200
    errors = np.zeros(max_iter)
    #alt_error = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    u_old = method(f, lam, **method_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = method(f, lam, **method_kwargs)
        errors[i] = np.linalg.norm(u_new - u_old, 'fro')
        t_list[i] = 1.0 / (1.0 + lam)

        #if(i == 1):
        #    min = errors[i]
        #else:
        #    if errors[i] < min:
        #        min = errors[i]
        #    else:
        #        break
        #alt_error[i] = np.linalg.norm(u_old - u_new) /abs(lam_init*(q ** i - q ** (i-1)))
        u_old = np.copy(u_new)
    if plot:
        plt.plot(t_list[1:], errors[1:])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$|\partial S(t)v/ \partial t|$')
        plt.grid()
        #plt.plot(alt_error)
        plt.show()
    opt_idx = np.argmax(errors[errors != 0.0])
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    print(t)
    lam = lam_init * (q ** opt_idx)
    u = method(f, lam, **method_kwargs)
    
    return u, t

def L_curve(f,method,lam_init = 1.0, q = 0.9, plot = True, **method_kwargs):
    """
    Implementation of the L-curve method
    """
    lam = lam_init
    max_iter = 75
    residual_list = np.zeros(max_iter)
    size_list = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    error = np.zeros(max_iter)
    for i in range(1, max_iter): #range(max_iter):
        print(i, lam)
        u = method(f, lam, **method_kwargs)
        lam = lam_init * (q ** i)
        t_list[i] = 1/(1 + lam)
        residual_list[i] = np.linalg.norm(u - f, 'fro')**2
        #size_list[i] = np.linalg.norm(norm1(grad(u)), 'fro')**2
        size_list[i] = np.sum(norm1(grad(u)))
        error[i]  = residual_list[i] * size_list[i]
        #if(i == 1):
        #    max = error[i]
        #else:
        #    if(error[i] > max):
        #        max = error[i]
        #    else:
        #        break
    if plot:
        plt.plot(np.log(size_list), np.log(residual_list))
        plt.xlabel(r'$|R(S(t)v|$')
        plt.ylabel(r'$|L(S(t)v,v)|$')
        plt.grid()
        plt.show()
        plt.plot(t_list[1:], error[error != 0.0])
        plt.xlabel(r'$t$')
        plt.ylabel(r'$|R(S(t)v||L(S(t)v,v)|$')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))#
        plt.grid()
        plt.show()
    opt_idx = np.argmax(error[error != 0.0])
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    print(t)
    lam = lam_init * (q ** opt_idx)
    u = method(f, lam, **method_kwargs)
    return u, t

def GCV(f,method,lam_init = 1.0, q = 0.9, **method_kwargs):
    """
    Implementation of the L-curve method
    """
    n = f.shape[0]
    lam = lam_init
    max_iter = 50
    GCV_list = np.zeros(max_iter)
    for i in range(1,max_iter): #range(max_iter):
        print(i, lam)
        w = numpy.random.normal(0,1,np.prod(f.shape)).reshape(f.shape)
        trace = np.sum(np.multiply(w, method(w, lam, **method_kwargs)))

        u_t = method(f, lam, **method_kwargs)
        GCV_list[i] = ((1/(n*n)) *  np.linalg.norm(u_t - f,'fro')**2)/(1 - (1/(n*n)) * trace)**2
        lam = lam_init * (q ** i)
    plt.plot(GCV_list)
    plt.show()
    opt_idx = np.argmin(GCV_list[GCV_list !=0.0])
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    lam = lam_init * (q ** opt_idx)
    u = method(f, lam, **method_kwargs)
    return u, t

def GCV_trace(f,method,lam_init = 1.0, q = 0.9, **method_kwargs):
    """

    """
    n = f.shape[0]
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    flap = fft2(laplacian, (n,n))
    lam = lam_init
    def trace(lam):
        ret = 0
        #eigs = fftn(1/(1 - lam* laplacian), (n,n))
        eigs = 1/(1 + lam*np.abs(flap))
        for i in range(n):
            for j in range(n):
                #ret+= 1 - 1/(1 + lam * eigs[i,j])
                ret+= 1 - eigs[i,j]
        return ret
    max_iter = 150
    GCV_list = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    for i in range(1,max_iter): #range(max_iter):
        print(i, lam)
        t_list[i] = 1/(1 + lam)
        u_t = method(f, lam, **method_kwargs)
        GCV_list[i] = n*n*np.linalg.norm(u_t - f,'fro')**2/trace(lam)**2
        lam = lam_init * (q ** i)
    plt.plot(t_list[1:], GCV_list[1:]/np.max(GCV_list))
    plt.grid()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$GCV(t)$')
    plt.show()
    opt_idx = np.argmin(GCV_list[GCV_list !=0.0])
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    print(t)
    lam = lam_init * (q ** opt_idx)
    u = method(f, lam, **method_kwargs)
    return u, t

def GCV_trace_convolution(f,method,fker, fker_star, shape,lam_init = 1.0, q = 0.9, **method_kwargs):
    """

    """
    n = f.shape[0]
    m = shape[0] - n +1
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    flap = rfftn(laplacian, shape)
    lam = lam_init
    def trace(lam):
        ret = 0
        #eigs = fftn(1/(1 - lam* laplacian), (n,n))
        eigs = np.abs(fker)**2/(np.abs(fker)**2 + lam*np.abs(flap))
        for i in range(n):
            for j in range(n//2):
                #ret+= 1 - 1/(1 + lam * eigs[i,j])
                ret+= 2*(1 - eigs[i,j])
        return ret
    max_iter = 20
    GCV_list = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    for i in range(1,max_iter): #range(max_iter):
        print(i, lam)
        t_list[i] = 1/(1 + lam)
        u_t = method(f, lam, fker, fker_star, shape, **method_kwargs)

        Au = center(irfftn(rfftn(edge_pad_and_shift(u_t,m),shape)*fker),n,m)
        GCV_list[i] = n*n*np.linalg.norm(Au - f,'fro')**2/trace(lam)**2
        lam = lam_init * (q ** i)
    plt.plot(t_list[1:], GCV_list[1:]/np.max(GCV_list))
    plt.grid()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$GCV(t)$')
    plt.show()
    opt_idx = np.argmin(GCV_list[GCV_list !=0.0])
    t = 1.0 / (1.0 + lam_init * (q ** opt_idx))
    print(t)
    lam = lam_init * (q ** opt_idx)
    u = method(f, lam, fker, fker_star, shape, **method_kwargs)
    return u, t

def R(t, f, u, method, u0 = None, **method_kwargs):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t = method(f, (1-t)/t, u0 = u0, **method_kwargs)
    return u_t, 0.5*np.linalg.norm(u_t - u)**2

def R_convolution(t,f,u,fker,fker_star,shape, method, **method_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1
    if (t == 1.0):
        u_t = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)/fker),n,m)
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t = method(f, (1-t)/t, fker, fker_star, shape, **method_kwargs)
    return u_t, 0.5* np.linalg.norm(u_t - u)**2

def R_basis(t, f, basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.25, acc = True, tol = 1e-4)
    return np.sum([np.sum(np.multiply(basis_img[i],u_t))**2 for i in range(basis_img.shape[0])]) 

def R_basis_vec(t, f, basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.2, acc = True, tol = 1e-2)
    shape = u_t.shape
    u_t = u_t.reshape(np.prod(shape)) 
    print(np.inner(basis_img, u_t)**2)
    return np.sum(np.inner(basis_img, u_t)**2) 

def R_basis_alt(t,f,avg_img,basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.2, acc = True, tol = 1e-2)
        u_t = u_t - avg_img
    return np.linalg.norm(u_t - np.sum([np.sum(np.multiply(basis_img[i],u_t))*basis_img[i] for i in range(basis_img.shape[0])]))**2

def R_basis_vec_alt(t,f,basis_img):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t =  ChambollePock_denoise(f, (1-t)/t, tau = 0.3, sig = 0.2, acc = True, tol = 1e-2)
    shape = u_t.shape
    u_t = u_t.reshape(np.prod(shape)) 
    return np.linalg.norm(u_t - np.sum([np.inner(basis_img[i],u_t)*basis_img[i] for i in range(basis_img.shape[0])]))

def gridSearch(f, u, N, method, plot=True, **method_kwargs):
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        _, R_list[i] = R(t_list[i],f,u, method,**method_kwargs)
        print(i, t_list[i], R_list[i])
    t_opt = t_list[np.argmin(R_list[1:N-1])]
    if plot:
        plt.plot(t_list[1:N-1],R_list[1:N-1], color= "red")
        #plt.plot(t_opt,np.min(R_list[1:N-1]), 'ro')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid()
        plt.show()
    return t_opt, R_list[1:]

def gridSearch_convolution(f, u, N, fker, fker_star, shape, method, plot=True, **method_kwargs):
    #t_list = np.linspace(0.99, 1.0, N)
    t_list = np.flip(1-np.logspace(-1.0, 0.0, N,base=10000000))
    R_list = np.zeros(N)
    for i in range(0,N):
        u_t, R_list[i] = R_convolution(t_list[i],f,u,fker,fker_star,shape, method, **method_kwargs)
        print(psnr(u_t,u))
        print(i, t_list[i], R_list[i])
        #R_listhat[i] = R2D(t_list[i],Y_test,X_hat)
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    #t_opthat = t_list[np.argmin(R_listhat[1:N-1])]
    if plot:
        plt.plot(t_list[1:N-1],R_list[1:N-1], color= "red")
        plt.plot(t_opttrue,np.min(R_list[1:N-1]), 'ro')
        #plt.plot(t_list[1:N-1],R_listhat[1:N-1],label="Empirical", color = "blue")
        #plt.plot(t_opthat,np.min(R_listhat[1:N-1]), 'bo')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid()
        plt.show()
    #print("optimal lambda", (1 - t_opttrue)/t_opttrue)
    return t_opttrue, R_list[1:N-1]

def gridSearch_basis(f, basis_img, avg_img, N):
    R_curr = lambda t: R_basis_alt(t,f,avg_img, basis_img)
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        R_list[i] = R_curr(t_list[i])
        print(i, t_list[i], R_list[i])
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    return t_opttrue, t_list[1:N-1], R_list[1:N-1] 

def gridSearch_basis_vec(f, basis_img, N):
    R_curr = lambda t: R_basis_vec(t,f,basis_img)
    t_list = np.linspace(0.85, 1.0, N)
    R_list = np.zeros(N)
    for i in range(0,N):
        R_list[i] = R_curr(t_list[i])
        print(i, t_list[i], R_list[i])
    t_opttrue = t_list[np.argmin(R_list[1:N-1])]
    return t_opttrue, t_list[1:N-1], R_list[1:N-1]

def optTV_dep(Y, X_hat, tol = 1e-5):
    """
    Implementation of backtracking line search with interpolation
    """
    
    h = - 0.001
    satisfied = False
    cnt = -1
    max_iter = 100
    beta = 0.8
    go = True
    alpha0 = 0.01
    tau = 0.45
    c1 = 1e-2
    t = np.zeros(max_iter + 1); t[0] = 0.99
    t_current = t[0]
    
    descent_directions = np.zeros(len(t))
    loss_functional_evals = np.zeros(len(t))
    alphas = np.zeros(len(t))
    
    while satisfied == False:
        # Compute gradient at t
        cnt += 1
        z, _ = ChambollePock_denoise(Y,(1 - t_current)/t_current)
        t_direction = t_current + h
        z_h, _ = ChambollePock_denoise(Y,(1 - t_direction)/t_direction)
        descent_directions[cnt] =  (np.linalg.norm(z - X_hat) - np.linalg.norm(z_h - X_hat)) / h
        loss_functional_evals[cnt] =  np.linalg.norm(z - X_hat)
    
        # Check stopping criteria
        if (np.abs(descent_directions[cnt]) < tol) or (cnt >= max_iter):
        	satisfied = True
        	break
    
        # Find the step-size
        alpha = np.copy(alpha0)
        tau = - np.abs(descent_directions[cnt]) ** 2
    
        if cnt == 0 :
            last_alpha = alpha
        t_temp = t_current + alpha * descent_directions[cnt]
        phi0 = loss_functional_evals[cnt]
        phiprime0 = tau
        z_temp, _ = ChambollePock_denoise(Y, (1 - t_temp) / t_temp)
        phialpha = np.linalg.norm(z_temp - X_hat)
    
        if (phialpha - phi0) < (c1 * phiprime0 * alpha):
            step_size = alpha
            # print "here and {0}, {1}, {2}, with finally {3}".format(phialpha - phi0, phi0, alpha, c1 * phiprime0 * alpha)
        else:
            step_size = - (alpha ** 2 * phiprime0) / (2.0 * (phialpha - phi0 - phiprime0 * alpha))
            # print "there {0}".format(step_size)
        if (np.abs(last_alpha / step_size) > 10.0) or (np.abs(step_size) < 1e-3):
            # print "{0} and {1}".format(np.abs(step_size - last_alpha), np.abs(step_size / last_alpha))
            step_size = last_alpha * beta * 1.3
    
        last_alpha = np.copy(step_size)
        t_new = t_current + step_size * descent_directions[cnt]
        t[cnt + 1] = t_new
        t_current = t_new
        print("cnt is ", cnt)
    return t_new

def optTV(f, u_hat, method, t_left = 0.0, t_right = 1.0, optTVtol = 1e-4, **method_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, R_left = R(t_left_new, f, u_hat, method, **method_kwargs)
    u, R_right = R(t_right_new, f, u_hat, method, **method_kwargs)

    i = 0
    while (i < max_iter):
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, R_left = R(t_left_new, f, u_hat, method, u0 = u, **method_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, R_right = R(t_right_new, f, u_hat, method, u0 = u,**method_kwargs)
        if (h < optTVtol):
            t_opt = (t_right_new + t_left_new)/2

            u = method(f, (1- t_opt)/t_opt, **method_kwargs)

            return u, t_opt
        i += 1

def optTV_convolution(f, u_hat, fker, fker_star,shape, method,t_left = 0.0, t_right = 1.0, tol = 1e-6, **method_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2
    
    R_ = lambda t, f, u: R_convolution(t,f,u,fker,fker_star,shape, method, **method_kwargs)

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    _, R_left = R_(t_left_new, f, u_hat)
    _, R_right = R_(t_right_new, f, u_hat)

    i = 0
    while (i < max_iter):
        print(t_left, t_right)
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            _, R_left = R_(t_left_new, f, u_hat)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            _, R_right = R_(t_right_new, f, u_hat)
        if (h < tol):
            t_opt = (t_right_new + t_left_new)/2

            u = method(f,(1-t_opt)/t_opt, fker, fker_star, shape, **method_kwargs) 

            return u, t_opt
        i += 1

def createEstimator(f, new_n, ds_number, solve_method, parameter_method, plot = False, **kwargs):
    n = f.shape[0]
    coords = np.zeros((2,n)) 
    coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1)

    ###
    # PRESOLVE f 
    ###

    coords_ds = np.zeros((ds_number**2,2,new_n))
    if (ds_number == 1):
        coords_ds[0,0,:] = np.linspace(0.5, n - 1.5, new_n, endpoint = True) 
        coords_ds[0,1,:] = np.linspace(0.5, n - 1.5, new_n, endpoint = True)
    else:
        for i in range(ds_number):
            for j in range(ds_number):
                coords_ds[ds_number*i + j,0,:] = np.linspace(i, n - 2 + i , new_n, endpoint = True) 
                coords_ds[ds_number*i + j,1,:] = np.linspace(j, n - 2 + j, new_n, endpoint = True)


    interpf = scipy.interpolate.interp2d(coords[0],coords[1],f, kind='linear')
    f_ds = np.zeros((new_n,new_n))
    f_us = np.zeros((ds_number**2,n,n))
    for i in range(ds_number**2):
        f_ds = interpf(coords_ds[i,0,:], coords_ds[i,1,:])
        if plot:
            plt.imsave('lenna_ds' + str(i) + '.png', f_ds, cmap= 'gray')
        ###
        # POSTSOLVE f
        ###

        if (solve_method == False or parameter_method == False):

            interpf_ds = scipy.interpolate.interp2d(coords_ds[i,0,:], coords_ds[i,1,:], f_ds, kind = 'linear')
        else:
            if(i == 0):
                u_ds, t_opt = parameter_method(f_ds,solve_method, **kwargs)
                print(t_opt)
            else:
                u_ds = solve_method(f_ds, (1-t_opt)/t_opt, **kwargs)

            interpf_ds = scipy.interpolate.interp2d(coords_ds[i,0,:], coords_ds[i,1,:], u_ds, kind = 'cubic')

        f_us[i] = interpf_ds(coords[0], coords[1])
    avg = np.average(f_us, axis = 0)
    return avg

def DP_func(f, t, method, u0 = None, noise_std = 0.0, param = 1.0, **method_kwargs):
    if (t == 1):
        u = f
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        if u0 is None:
            u = method(f, (1-t)/t, **method_kwargs) 
        else:
            u = method(f, (1-t)/t, u0 = u0, **method_kwargs)
    return u, np.linalg.norm(u - f) - param * u.shape[0] * noise_std

def DP_func_convolution(f, t, method, fker, fker_star, shape, noise_std = 0.0, param = 1.0, **method_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1

    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(f,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = method(f, (1-t)/t, fker, fker_star, shape, **method_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    return u, np.linalg.norm(Au - f) - param * u.shape[0]* noise_std

def DP_bisection(f, method, noise_std = 0.0, param = 1.0, t_left = 0.5, t_right = 1.0,DP_tol1 = 1.0e-4,DP_tol2 = 1.0e-5, **method_kwargs):
    max_iter = 100

    u, DP_left = DP_func(f, t_left, method,  noise_std = noise_std, param = param, **method_kwargs)
    for i in range(max_iter):
        t_mid = (t_left + t_right)/2
        if (t_right - t_left)/2 > DP_tol2:
            u, DP_mid = DP_func(f, t_mid, method, u0 = u, noise_std = noise_std, param = param, **method_kwargs)
            if np.abs(DP_mid) < DP_tol1:
                return u, t_mid
            else:
                if np.sign(DP_mid) == np.sign(DP_left):
                    t_left = t_mid
                    DP_left = DP_mid
                else:
                    t_right = t_mid
        else:
            return u, t_mid

def DP_rf(f, method, noise_std = 0.0, param = 1.0, t_left = 0.0, t_right = 1.0,DP_tol1 = 1.0e-3,DP_tol2 = 1.0e-4, **method_kwargs):
    max_iter = 100
    u, DP_left = DP_func(f, t_left, method, noise_std, param, **method_kwargs)
    u, DP_right = DP_func(f, t_right, method, noise_std, param, **method_kwargs)
    for i in range(max_iter):
        t_mid = (t_left*DP_right - t_right*DP_left)/(DP_right - DP_left)
        if (t_right - t_left)/2 > DP_tol2:
            u, DP_mid = DP_func(f, t_mid, method, noise_std, param, **method_kwargs)
            print(DP_mid)
            if np.abs(DP_mid) < DP_tol1:
                print( np.abs(DP_mid))
                return u, t_mid
            else:
                if np.sign(DP_mid) == np.sign(DP_left):
                    t_left = t_mid
                    DP_left = DP_mid
                else:
                    t_right = t_mid
                    DP_right = DP_mid
        else:
            return u, t_mid

def DP_secant(f, method, noise_std = 0.0, param = 1.0, t_0 = 1.0, t_1 = 0.99, DP_tol = 1.0e-4, **method_kwargs):
    max_iter = 100
    u, DP_pp = DP_func(f, t_0, method, noise_std = noise_std, param = param, **method_kwargs)
    u, DP_p = DP_func(f, t_1, method, noise_std = noise_std, param = param, **method_kwargs)
    t_pp = t_0
    t_p = t_1
    for i in range(max_iter):
        t_c = t_p - DP_p*(t_p - t_pp)/(DP_p - DP_pp)
        #u, DP_c = DP_func(f, t_c, method, u0 = u, noise_std = noise_std, param = param, **method_kwargs)
        u, DP_c = DP_func(f, t_c, method, u0 = u, noise_std = noise_std, param = param, **method_kwargs)
        if np.abs(DP_c)/f.shape[0] < DP_tol:
            return u, t_c
        else:
            t_pp = t_p
            t_p = t_c
            DP_pp = DP_p
            DP_p = DP_c

def DP_secant_convolution(f, method, fker, fker_star, shape, noise_std = 0.0, param = 1.1, t_0 = 1.0, t_1 = 0.99, DP_tol = 1.0e-8, **method_kwargs):
    max_iter = 100
    u, DP_pp = DP_func_convolution(f, t_0, method, fker, fker_star, shape, noise_std = noise_std, param = param, **method_kwargs)
    u, DP_p = DP_func_convolution(f, t_1, method, fker, fker_star, shape, noise_std = noise_std, param = param, **method_kwargs)
    t_pp = t_0
    t_p = t_1
    for i in range(max_iter):
        t_c = t_p - DP_p*(t_p - t_pp)/(DP_p - DP_pp)
        u, DP_c = DP_func_convolution(f, t_c, method, fker, fker_star, shape,noise_std = noise_std, param = param, **method_kwargs)
        if np.abs(DP_c)/f.shape[0] < DP_tol:
            return u, t_c
        else:
            t_pp = t_p
            t_p = t_c
            DP_pp = DP_p
            DP_p = DP_c

def QOC(f, method, lam_init = 1.0, q = 0.9, break_iter = 5, **method_kwargs):
    max_iter = 100
    u_old = method(f, lam_init, **method_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = method(f, lam, **method_kwargs)
        loss = np.linalg.norm(u_new - u_old, 'fro')
        if(i == 1):
            minloss = loss
            loss_i = i
        else:
            if loss < minloss:
                minloss = loss
                loss_i = i
            else:
                if (i>= loss_i + break_iter):
                    break
        u_old = np.copy(u_new)
    lam = lam_init * (q ** loss_i)
    u = method(f, lam, **method_kwargs)

    return u, 1/(1 + lam)

def QOC_convolution(f, method, fker, fker_star, shape, lam_init = 1.0, q = 0.9, break_iter = 5, **method_kwargs):
    max_iter = 100
    u_old = method(f, lam_init, fker, fker_star, shape, **method_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = method(f, lam,fker, fker_star, shape,  **method_kwargs)
        loss = np.linalg.norm(u_new - u_old, 'fro')
        print(lam, loss)
        if(i == 1):
            minloss = loss
            loss_i = i
        else:
            if loss < minloss:
                minloss = loss
                loss_i = i
            else:
                if (i>= loss_i + break_iter):
                    break
        u_old = np.copy(u_new)
    lam = lam_init * (q ** loss_i)
    u = method(f, lam,fker,fker_star,shape, **method_kwargs)

    return u, 1/(1 + lam)

def LC_func(t, f, method, u0 = None, name='TV',**method_kwargs):
    if (t == 1):
        u = f
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        if u0 is None:
            u = method(f, (1-t)/t, **method_kwargs) 
        else:
            u = method(f, (1-t)/t, u0 = u0, **method_kwargs)
    if name=='TV':
        return u, -np.linalg.norm(u - f, 'fro')**2 * np.sum(norm1(grad(u)))
    elif name == 'QV':
        return u, -np.linalg.norm(u - f, 'fro')**2 *np.linalg.norm(norm1(grad(u)), 'fro')

def LC_func_convolution(t, f, method, fker, fker_star, shape, name,**method_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1
    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = method(f, (1-t)/t, fker, fker_star, shape, **method_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    if name == 'QV':
        return u, -np.linalg.norm(u - f, 'fro')**2 *np.linalg.norm(norm1(grad(u)), 'fro')
    elif name == 'TV':
        return u, -np.linalg.norm(u - f, 'fro')**2 * np.sum(norm1(grad(u)))

def LC_golden(f, method, t_left = 0.8, t_right = 1.0, goldentol = 1e-3, name = 'TV',**method_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, LC_left = LC_func(t_left_new, f, method, name=name,**method_kwargs)
    u, LC_right = LC_func(t_right_new, f, method, name=name,**method_kwargs)

    i = 0
    while (i < max_iter):
        if (LC_left < LC_right):
            t_right = t_right_new
            t_right_new = t_left_new
            LC_right = LC_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, LC_left = LC_func(t_left_new, f, method, u0 = u, name=name,**method_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            LC_left = LC_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, LC_right = LC_func(t_right_new, f, method, u0 = u,name=name,**method_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = method(f, (1- t_opt)/t_opt, **method_kwargs)

            return u, t_opt
        i += 1

def LC_golden_convolution(f, method, fker, fker_star, shape, t_left = 0.8, t_right = 1.0, goldentol = 1e-5, name = 'TV', **method_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, LC_left = LC_func_convolution(t_left_new, f, method, fker, fker_star, shape,name,**method_kwargs)
    u, LC_right = LC_func_convolution(t_right_new, f, method, fker, fker_star, shape,name,**method_kwargs)

    i = 0
    while (i < max_iter):
        if (LC_left < LC_right):
            t_right = t_right_new
            t_right_new = t_left_new
            LC_right = LC_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, LC_left = LC_func_convolution(t_left_new, f, method, fker, fker_star, shape, name, **method_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            LC_left = LC_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, LC_right = LC_func_convolution(t_right_new, f, method, fker,fker_star,shape,name,**method_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = method(f, (1- t_opt)/t_opt, fker, fker_star, shape, **method_kwargs)

            return u, t_opt
        i += 1

def GCV_func(t, f, trace, method, u0 = None, **method_kwargs):
    
    if (t == 1):
        u = f
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        if u0 is None:
            u = method(f, (1-t)/t, **method_kwargs) 
        else:
            u = method(f, (1-t)/t, u0 = u0, **method_kwargs)
    return u, np.linalg.norm(u - f,'fro')**2/trace((1-t)/t)**2

def GCV_func_convolution(t, f, trace, method, fker, fker_star, shape, **method_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1
    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = method(f, (1-t)/t, fker, fker_star, shape, **method_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    return u, np.linalg.norm(Au - f,'fro')**2/trace((1-t)/t)**2


def GCV_golden(f, method, t_left = 0.0, t_right = 1.0 - 1e-4, goldentol = 1e-3, **method_kwargs):
    def trace_ker(lam, fker):
        ret = 0
        eigs = 1/(1 + lam*np.abs(fker))
        for i in range(n):
            for j in range(n//2):
                ret+= 1 - eigs[i,j]
        return ret
    n = f.shape[0]
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    fker = rfftn(laplacian, (n,n))
    trace = lambda lam : trace_ker(lam,fker) 
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, GCV_left = GCV_func(t_left_new, f, trace, method, **method_kwargs)
    u, GCV_right = GCV_func(t_right_new, f, trace, method, **method_kwargs)

    i = 0
    while (i < max_iter):
        if (GCV_left < GCV_right):
            t_right = t_right_new
            t_right_new = t_left_new
            GCV_right = GCV_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, GCV_left = GCV_func(t_left_new, f, trace, method, u0 = u, **method_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            GCV_left = GCV_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, GCV_right = GCV_func(t_right_new, f, trace, method, u0 = u,**method_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = method(f, (1- t_opt)/t_opt, **method_kwargs)

            return u, t_opt
        i += 1
def GCV_golden_convolution(f, method, fker, fker_star, shape, t_left = 0.0, t_right = 1.0 - 1e-4, goldentol = 1e-3, **method_kwargs):
    def trace_ker(lam, fker, flap):
        ret = 0
        eigs = np.abs(fker)**2/(np.abs(fker)**2 + lam*np.abs(flap))
        for i in range(n):
            for j in range(n//2):
                ret+= 1 - eigs[i,j]
        return ret
    n = f.shape[0]
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    flap = rfftn(laplacian, shape)
    trace = lambda lam : trace_ker(lam,fker,flap) 
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, GCV_left = GCV_func_convolution(t_left_new, f, trace, method, fker, fker_star, shape,**method_kwargs)
    u, GCV_right = GCV_func_convolution(t_right_new, f, trace, method, fker, fker_star, shape,**method_kwargs)

    i = 0
    while (i < max_iter):
        if (GCV_left < GCV_right):
            t_right = t_right_new
            t_right_new = t_left_new
            GCV_right = GCV_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, GCV_left = GCV_func_convolution(t_left_new, f, trace, method, fker, fker_star, shape, **method_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            GCV_left = GCV_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, GCV_right = GCV_func_convolution(t_right_new, f, trace, method, fker, fker_star, shape, **method_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = method(f, (1- t_opt)/t_opt, fker, fker_star, shape,**method_kwargs)

            return u, t_opt
        i += 1



