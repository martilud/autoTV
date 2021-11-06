import numpy as np
import scipy as sp
import numpy.random
import matplotlib.pyplot as plt
import time
import scipy.interpolate
from utils import *
from numpy.fft import fftn

def discrepancy_rule(f,solver, noise_std = 0.0, param = 1.0, lam_init= 1.0, q = 0.90, plot = False, **solver_kwargs):
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
        u = solver(f, lam, **solver_kwargs)
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

def discrepancy_rule_convolution(f,solver, fker, fker_star, shape, noise_std = 0.0, param = 1.0, lam_init= 0.01, q = 0.95, plot = False, **solver_kwargs):
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
        u = solver(f, lam, fker, fker_star, shape,**solver_kwargs)
        lam = lam_init * (q ** i)
        if plot:
            t_list[i] = 1.0 / (1.0 + lam_init * (q ** i))
            print(t_list[i])
            Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
            disp_list[i] = np.linalg.norm(Au-f)
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

def monotone_rule(f, solver, noise_std = 0.0, param = 1.0, lam_init= 2.0, q = 0.95, **solver_kwargs):
    """
    Implementation of the discrepancy principle
    noise_std is the estimate (or the true value) of the noise parameter
    """
    nn = np.prod(f.shape)
    lam = lam_init
    max_iter = 50
    error_level = param * noise_std * np.sqrt(nn)
    me_tol = 1e-5
    u_old = solver(f, lam, **solver_kwargs)
    i = 1
    list = np.zeros(max_iter)
    while (i < max_iter):
        print(i)
        lam = lam_init * (q ** i)
        u_new = solver(f, lam, **solver_kwargs)
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

def quasi_optimality(f, solver,lam_init = 1.0, q = 0.90, plot = False, **solver_kwargs):
    """
    Implementation of quasi optimality for parameter selection
    """
    
    lam = lam_init
    max_iter = 200
    errors = np.zeros(max_iter)
    #alt_error = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    u_old = solver(f, lam, **solver_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = solver(f, lam, **solver_kwargs)
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
    u = solver(f, lam, **solver_kwargs)
    
    return u, t

def L_curve(f,solver,lam_init = 1.0, q = 0.9, plot = True, **solver_kwargs):
    """
    Implementation of the L-curve solver
    """
    lam = lam_init
    max_iter = 75
    residual_list = np.zeros(max_iter)
    size_list = np.zeros(max_iter)
    t_list = np.zeros(max_iter)
    error = np.zeros(max_iter)
    for i in range(1, max_iter): #range(max_iter):
        print(i, lam)
        u = solver(f, lam, **solver_kwargs)
        lam = lam_init * (q ** i)
        t_list[i] = 1/(1 + lam)
        residual_list[i] = np.linalg.norm(u - f, 'fro')**2
        size_list[i] = np.sum(norm1(grad(u)))
        error[i]  = residual_list[i] * size_list[i]
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
    u = solver(f, lam, **solver_kwargs)
    return u, t

def GCV_trace(f,solver,lam_init = 1.0, q = 0.9, **solver_kwargs):
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
        u_t = solver(f, lam, **solver_kwargs)
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
    u = solver(f, lam, **solver_kwargs)
    return u, t

def GCV_trace_convolution(f,solver,fker, fker_star, shape,lam_init = 1.0, q = 0.9, **solver_kwargs):
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
        u_t = solver(f, lam, fker, fker_star, shape, **solver_kwargs)

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
    u = solver(f, lam, fker, fker_star, shape, **solver_kwargs)
    return u, t

def R(t, f, u, solver, u0 = None, **solver_kwargs):
    if (t == 1.0):
        u_t = f
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t = solver(f, (1-t)/t, u0 = u0, **solver_kwargs)
    return u_t, 0.5*np.linalg.norm(u_t - u)**2

def R_convolution(t,f,u,fker,fker_star,shape, solver, **solver_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1
    if (t == 1.0):
        u_t = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)/fker),n,m)
    elif (t == 0.0):
        u_t = np.full(f.shape, np.average(f))
    else:
        u_t = solver(f, (1-t)/t, fker, fker_star, shape, **solver_kwargs)
    return u_t, 0.5* np.linalg.norm(u_t - u)**2


def gridSearch(A, v, u_hat, N, solver, plot=True, **solver_kwargs):
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    def R_(t, A, v, u, solver, **solver_kwargs):
        u_t = solver(A,v,t,1-t, **solver_kwargs)
        return u_t, np.linalg.norm(u_t - u_hat)**2
    u = solver(A,v,0.0,1.0, **solver_kwargs)
    for i in range(1,N-1):
        u, R_list[i] = R_(t_list[i],A,v,u_hat,solver,u0 = u,**solver_kwargs)
    t_opt = t_list[np.argmin(R_list[1:N-1])]
    if plot:
        plt.semilogy(t_list[1:N-1],R_list[1:N-1], color= "red")
        #plt.plot(t_opt,np.min(R_list[1:N-1]), 'ro')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid()
        plt.show()
    return t_opt, R_list[1:-1]


def golden_section(func, A, v, u_hat, solver, t_left = 0.000001, t_right = 0.9999999, gstol = 1e-4, gsmax_iter = 100, **kwargs):
    """
    Golden section search for optimizing functions of one variable, namely
    the regularization parameter.

    Parameters  :
        func    : Function to optimize. Should take in parameters on the form  
        A, v, lam_l, lam_r, solver, noise_std
        and return the resulting regularized solution and the value of func

    """

    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2
    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, f_left = func(t_left_new, A, v, u_hat, solver, noise_std, **solver_kwargs)
    u, f_right = func(t_right_new, A, v, u_hat, solver, noise_std, **solver_kwargs)

    i = 0
    while (i < gsmax_iter):
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, R_left = R_(t_left_new, A, v, u_hat, solver, u0 = u, **solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, R_right = R_(t_right_new, A, v, u_hat, solver, u0 = u,**solver_kwargs)
        if (h < gstol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(A, v, t_opt, 1-t_opt, u0 = u, **solver_kwargs)

            return u, t_opt
        i += 1
def optTV(f, u_hat, solver, t_left = 0.5, t_right = 0.99, optTVtol = 1e-4, **solver_kwargs):
    """
    CHANGED SO THAT solver TAKES IN t, NOT LAMBDA
    """
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, R_left = R(t_left_new, f, u_hat, solver, **solver_kwargs)
    u, R_right = R(t_right_new, f, u_hat, solver, **solver_kwargs)

    i = 0
    while (i < max_iter):
        if (R_left < R_right):
            t_right = t_right_new
            t_right_new = t_left_new
            R_right = R_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, R_left = R(t_left_new, f, u_hat, solver, u0 = u, **solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            R_left = R_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, R_right = R(t_right_new, f, u_hat, solver, u0 = u,**solver_kwargs)
        if (h < optTVtol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(f, (1-t_opt)/t_opt, **solver_kwargs)

            return u, t_opt
        i += 1

def optTV_convolution(f, u_hat, fker, fker_star,shape, solver,t_left = 0.0, t_right = 1.0, tol = 1e-6, **solver_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2
    
    R_ = lambda t, f, u: R_convolution(t,f,u,fker,fker_star,shape, solver, **solver_kwargs)

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

            u = solver(f,(1-t_opt)/t_opt, fker, fker_star, shape, **solver_kwargs) 

            return u, t_opt
        i += 1

def createEstimator(f, new_n, ds_number, solve_solver, parameter_solver, plot = False, **kwargs):
    n = f.shape[0]
    coords = np.zeros((2,n)) 
    coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1)

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

        if (solve_solver == False or parameter_solver == False):

            interpf_ds = scipy.interpolate.interp2d(coords_ds[i,0,:], coords_ds[i,1,:], f_ds, kind = 'linear')
        else:
            if(i == 0):
                u_ds, t_opt = parameter_solver(f_ds,solve_solver, **kwargs)
                print(t_opt)
            else:
                u_ds = solve_solver(f_ds, (1-t_opt)/t_opt, **kwargs)

            interpf_ds = scipy.interpolate.interp2d(coords_ds[i,0,:], coords_ds[i,1,:], u_ds, kind = 'cubic')

        f_us[i] = interpf_ds(coords[0], coords[1])
    avg = np.average(f_us, axis = 0)
    return avg

def DP_func(A,v, lam_l, lam_r, solver, noise_std = 0.0, param = 1.0, u0 = None, **solver_kwargs):
    u = solver(A,v, lam_l,lam_r, u0 = u0, **solver_kwargs)
    return u, np.linalg.norm(np.dot(A,u) - v) - param * np.sqrt(np.prod(v.shape)) * noise_std

def DP_func_convolution(f, t, solver, fker, fker_star, shape, noise_std = 0.0, param = 1.0, **solver_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1

    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(f,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = solver(f, (1-t)/t, fker, fker_star, shape, **solver_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    return u, np.linalg.norm(Au - f) - param * u.shape[0]* noise_std

def DP_secant(A, v, solver, noise_std = 0.0, param = 1.0, t_0 = 1.0, t_1 = 0.99999, dptol = 1.0e-8, dpmaxiter = 100, **solver_kwargs):
    if 'method' in solver_kwargs:
        if solver_kwargs['method'] == 'eigh':
            if not 'eigvals' in solver_kwargs and not 'eigs' in solver_kwargs:
                eigvals, eigs = sp.linalg.eigh(np.matmul(A.T,A))
                solver_kwargs['eigvals'] = eigvals
                solver_kwargs['eigs'] = eigs

    # Calculate function values in the two starting values t_0 and t_1
    u, DP_pp = DP_func(A, v, t_0, 1.0 - t_0, solver, noise_std = noise_std, param = param, **solver_kwargs)
    u, DP_p = DP_func(A, v , t_1, 1.0 - t_1, solver, noise_std = noise_std, param = param, u0 = u,**solver_kwargs)
    t_pp = t_0
    t_p = t_1
    for i in range(dpmaxiter):
        t_c = t_p - DP_p*(t_p - t_pp)/(DP_p - DP_pp)
        print(t_c)
        u, DP_c = DP_func(A,v, t_c, 1.0 - t_c, solver, noise_std = noise_std, param = param, u0 = u,**solver_kwargs)
        if np.abs(DP_c)/np.sqrt(u.shape[0]) < dptol:
            return u, t_c
        else:
            t_pp = t_p
            t_p = t_c
            DP_pp = DP_p
            DP_p = DP_c
    return u,t_c

def DP_bisection(A, v, solver, noise_std = 0.0, param = 1.0, t_left = 0.0, t_right = 1.0,DP_tol1 = 1.0e-4,DP_tol2 = 1.0e-5, max_iter = 100, **solver_kwargs):
    if 'method' in solver_kwargs:
        if solver_kwargs['method'] == 'eigh':
            if not 'eigvals' in solver_kwargs and not 'eigs' in solver_kwargs:
                eigvals, eigs = sp.linalg.eigh(np.matmul(A.T,A))
                solver_kwargs['eigvals'] = eigvals
                solver_kwargs['eigs'] = eigs

    u, DP_left = DP_func(A,v, t_left, 1.0 - t_left, solver, noise_std = noise_std, param = param, **solver_kwargs)
    for i in range(max_iter):
        t_mid = (t_left + t_right)/2
        if (t_right - t_left)/2 > DP_tol2:
            u, DP_mid = DP_func(A,v, t_mid,1.0 - t_mid, solver, u0 = u, noise_std = noise_std, param = param, **solver_kwargs)
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

def DP_secant_convolution(f, solver, fker, fker_star, shape, noise_std = 0.0, param = 1.1, t_0 = 1.0, t_1 = 0.99, DP_tol = 1.0e-8, **solver_kwargs):
    max_iter = 100
    u, DP_pp = DP_func_convolution(f, t_0, solver, fker, fker_star, shape, noise_std = noise_std, param = param, **solver_kwargs)
    u, DP_p = DP_func_convolution(f, t_1, solver, fker, fker_star, shape, noise_std = noise_std, param = param, **solver_kwargs)
    t_pp = t_0
    t_p = t_1
    for i in range(max_iter):
        t_c = t_p - DP_p*(t_p - t_pp)/(DP_p - DP_pp)
        u, DP_c = DP_func_convolution(f, t_c, solver, fker, fker_star, shape,noise_std = noise_std, param = param, **solver_kwargs)
        if np.abs(DP_c)/f.shape[0] < DP_tol:
            return u, t_c
        else:
            t_pp = t_p
            t_p = t_c
            DP_pp = DP_p
            DP_p = DP_c
    return u,t_c

def QOC(f, solver, lam_init = 1.0, q = 0.9, break_iter = 5, **solver_kwargs):
    max_iter = 100
    u_old = solver(f, lam_init, **solver_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = solver(f, lam, **solver_kwargs)
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
    u = solver(f, lam, **solver_kwargs)

    return u, 1/(1 + lam)

def QOC_convolution(f, solver, fker, fker_star, shape, lam_init = 1.0, q = 0.9, break_iter = 5, **solver_kwargs):
    max_iter = 100
    u_old = solver(f, lam_init, fker, fker_star, shape, **solver_kwargs)
    for i in range(1, max_iter):
        lam = lam_init * (q ** i)
        u_new = solver(f, lam,fker, fker_star, shape,  **solver_kwargs)
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
    u = solver(f, lam,fker,fker_star,shape, **solver_kwargs)

    return u, 1/(1 + lam)

def LC_func(t, f, solver, u0 = None, name='TV',**solver_kwargs):
    u = solver(f, (1-t)/t, u0 = u0, **solver_kwargs)
    if name=='TV':
        return u, -np.linalg.norm(u - f, 'fro')**2 * np.sum(norm1(grad(u)))
    elif name == 'QV':
        return u, -np.linalg.norm(u - f, 'fro')**2 * np.linalg.norm(norm1(grad(u)), 'fro')

def LC_func_convolution(t, f, solver, fker, fker_star, shape, name,**solver_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1
    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = solver(f, (1-t)/t, fker, fker_star, shape, **solver_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    if name == 'QV':
        return u, -np.linalg.norm(u - f, 'fro')**2 *np.linalg.norm(norm1(grad(u)), 'fro')
    elif name == 'TV':
        return u, -np.linalg.norm(u - f, 'fro')**2 * np.sum(norm1(grad(u)))

def LC_golden(f, solver, t_left = 0.8, t_right = 1.0, goldentol = 1e-3, name = 'TV',**solver_kwargs):

    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, LC_left = LC_func(t_left_new, f, solver, name=name,**solver_kwargs)
    u, LC_right = LC_func(t_right_new, f, solver, name=name,**solver_kwargs)

    i = 0
    while (i < max_iter):
        if (LC_left < LC_right):
            t_right = t_right_new
            t_right_new = t_left_new
            LC_right = LC_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, LC_left = LC_func(t_left_new, f, solver, u0 = u, name=name,**solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            LC_left = LC_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, LC_right = LC_func(t_right_new, f, solver, u0 = u,name=name,**solver_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(f, (1- t_opt)/t_opt, **solver_kwargs)

            return u, t_opt
        i += 1

def LC_golden_convolution(f, solver, fker, fker_star, shape, t_left = 0.8, t_right = 1.0, goldentol = 1e-5, name = 'TV', **solver_kwargs):
    max_iter = 100
    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    #n = int(np.ceil(np.log(tol / h) / np.log(phi)))
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, LC_left = LC_func_convolution(t_left_new, f, solver, fker, fker_star, shape,name,**solver_kwargs)
    u, LC_right = LC_func_convolution(t_right_new, f, solver, fker, fker_star, shape,name,**solver_kwargs)

    i = 0
    while (i < max_iter):
        if (LC_left < LC_right):
            t_right = t_right_new
            t_right_new = t_left_new
            LC_right = LC_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, LC_left = LC_func_convolution(t_left_new, f, solver, fker, fker_star, shape, name, **solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            LC_left = LC_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, LC_right = LC_func_convolution(t_right_new, f, solver, fker,fker_star,shape,name,**solver_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(f, (1- t_opt)/t_opt, fker, fker_star, shape, **solver_kwargs)

            return u, t_opt
        i += 1

def GCV_func(A, v, lam_l, lam_r, solver, u0 = None, **solver_kwargs):
    u = solver(A,v, lam_l, lam_r, **solver_kwargs) 
    trace = np.sum(lam_l * solver_kwargs['eigvals']/(lam_l * solver_kwargs['eigvals'] + lam_r)) 
    return u, np.linalg.norm(np.dot(A,u) - v)**2/(1 - trace/np.prod(v.shape))**2
    #return u, np.linalg.norm(np.dot(A,u) - v)**2 + 2 * 0.1 * trace

def GCV_func_convolution(t, f, trace, solver, fker, fker_star, shape, **solver_kwargs):
    n = f.shape[0]
    m = shape[0] - n + 1
    if (t == 1):
        u = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)/fker),n,m)
    elif (t == 0):
        u = np.full(f.shape, np.average(f))
    else:
        u = solver(f, (1-t)/t, fker, fker_star, shape, **solver_kwargs)
    Au = center(irfftn(rfftn(edge_pad_and_shift(u,m),shape)*fker),n,m)
    return u, np.linalg.norm(Au - f,'fro')**2/trace((1-t)/t)**2


def GCV_golden(A, v, solver, t_left = 0.0, t_right = 1.0, goldentol = 1e-4, **solver_kwargs):
    max_iter = 100

    if 'method' in solver_kwargs:
        if solver_kwargs['method'] == 'eigh':
            if not 'eigvals' in solver_kwargs and not 'eigs' in solver_kwargs:
                eigvals, eigs = sp.linalg.eigh(np.matmul(A.T,A))
                solver_kwargs['eigvals'] = eigvals
                solver_kwargs['eigs'] = eigs

    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, GCV_left = GCV_func(A, v, t_left_new, 1.0 - t_left_new, solver, **solver_kwargs)
    u, GCV_right = GCV_func(A, v, t_right_new,1.0 - t_right_new, solver, **solver_kwargs)

    i = 0
    while (i < max_iter):
        if (GCV_left < GCV_right):
            t_right = t_right_new
            t_right_new = t_left_new
            GCV_right = GCV_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, GCV_left = GCV_func(A, v, t_left_new, 1.0 - t_left_new, solver, u0 = u,**solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            GCV_left = GCV_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, GCV_right = GCV_func(A, v, t_right_new,1.0 - t_right_new, solver, u0 = u, **solver_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(A,v, t_opt, 1.0 - t_opt, **solver_kwargs)

            return u, t_opt
        i += 1
def UPRE_func(A, v, lam_l, lam_r, sig_noise, solver, u0 = None, **solver_kwargs):
    u = solver(A,v, lam_l, lam_r, **solver_kwargs) 
    trace = np.sum(lam_l * solver_kwargs['eigvals']/(lam_l * solver_kwargs['eigvals'] + lam_r)) 
    return u, np.linalg.norm(np.dot(A,u) - v)**2 + 2 * sig_noise * trace

def UPRE_golden(A, v, solver, sig_noise = 0.0, t_left = 0.0, t_right = 1.0, goldentol = 1e-4, **solver_kwargs):
    max_iter = 100

    if 'method' in solver_kwargs:
        if solver_kwargs['method'] == 'eigh':
            if not 'eigvals' in solver_kwargs and not 'eigs' in solver_kwargs:
                eigvals, eigs = sp.linalg.eigh(np.matmul(A.T,A))
                solver_kwargs['eigvals'] = eigvals
                solver_kwargs['eigs'] = eigs

    rho = (3 - np.sqrt(5))/2
    phi = (np.sqrt(5) -1)/2

    h = t_right - t_left
    t_left_new = t_left + rho * h
    t_right_new = t_left + phi * h
    t_right_new = t_left + phi * h
    u, UPRE_left = UPRE_func(A, v, t_left_new, 1.0 - t_left_new, sig_noise, solver, **solver_kwargs)
    u, UPRE_right = UPRE_func(A, v, t_right_new,1.0 - t_right_new, sig_noise, solver, **solver_kwargs)

    i = 0
    while (i < max_iter):
        if (UPRE_left < UPRE_right):
            t_right = t_right_new
            t_right_new = t_left_new
            UPRE_right = UPRE_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, UPRE_left = UPRE_func(A, v, t_left_new, 1.0 - t_left_new, sig_noise, solver, u0 = u,**solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            UPRE_left = UPRE_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, UPRE_right = UPRE_func(A, v, t_right_new,1.0 - t_right_new, sig_noise, solver, u0 = u, **solver_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(A,v, t_opt, 1.0 - t_opt, **solver_kwargs)

            return u, t_opt
        i += 1

def GCV_golden_convolution(f, solver, fker, fker_star, shape, t_left = 0.0, t_right = 1.0 - 1e-4, goldentol = 1e-3, **solver_kwargs):
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
    u, GCV_left = GCV_func_convolution(t_left_new, f, trace, solver, fker, fker_star, shape,**solver_kwargs)
    u, GCV_right = GCV_func_convolution(t_right_new, f, trace, solver, fker, fker_star, shape,**solver_kwargs)

    i = 0
    while (i < max_iter):
        if (GCV_left < GCV_right):
            t_right = t_right_new
            t_right_new = t_left_new
            GCV_right = GCV_left
            h = phi * h
            t_left_new = t_left + rho * h
            u, GCV_left = GCV_func_convolution(t_left_new, f, trace, solver, fker, fker_star, shape, **solver_kwargs)
        else:
            t_left = t_left_new
            t_left_new = t_right_new
            GCV_left = GCV_right
            h = phi * h
            t_right_new = t_left + phi * h
            u, GCV_right = GCV_func_convolution(t_right_new, f, trace, solver, fker, fker_star, shape, **solver_kwargs)
        if (h < goldentol):
            t_opt = (t_right_new + t_left_new)/2

            u = solver(f, (1- t_opt)/t_opt, fker, fker_star, shape,**solver_kwargs)

            return u, t_opt
        i += 1

