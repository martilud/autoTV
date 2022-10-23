from os import error
import autograd.numpy as np
#from utils import *
import scipy as sp
from problem import *
from autograd import grad
import matplotlib.pyplot as plt
from parameterselection import opt

#D = D_zero_boundary
#Dast = Dast_zero_boundary

def py_ridge(prob, v, lam_l, lam_r, 
    Av = None, u0 = None):
    """
    Solver for the so-called ridge problem min_u lam_l \|Au - v\|^2 + lam_r/2 \|u\|^2
    A, u and v are more commonly denoted as X, beta and Y in regression litterature.
    Parameters  :
        A : (N,M) ndarray. Contains the linear operator.
        v : (N) ndarray
        lam_l : float. "Left" regularization parameter
        lam_r : float. "Right" regularization parameter
        solve_method : string. 'chol', 'eigh' or 'solve',
            what method is used to solve the ridge problem.
        eigvals : (M) ndarray. Eigenvalues of A^TA.
        eigs : (M,M) ndarray. Orthonormal eigenvectors of A^T A.
        Av : (M) ndarray. A^Tv.
        u0 : (M) ndarray. Initial value, unused for this implementation.
    Returns :
        u : (M) ndarray. Solution to ridge problem  
    """
    if lam_l == 0 and lam_r > 0:
        return np.zeros(prob.A.shape[1])
    elif lam_l < 0 or lam_r < 0 or (lam_l == 0 and lam_r == 0):
        raise RuntimeError('py_ridge Requires at least one positive regularization parameter lam_l and lam_r') 
        
    if Av is None:
        rhs = lam_l * np.dot(prob.A.T, v)
    else:
        rhs = lam_l * Av

    if prob.solve_method == 'chol': 
        lhs = (lam_l * prob.AA + lam_r * np.identity(prob.A.shape[1]))
        chol = sp.linalg.cho_factor(lhs)
        u = sp.linalg.cho_solve(chol,rhs)
    elif prob.solve_method == 'eigh':
        u = np.dot(np.multiply(prob.eigs, 1.0/(lam_l * prob.eigvals + lam_r)), np.dot(prob.eigs.T.conj(), rhs))
    elif prob.solve_method == 'svd':
        u = np.dot(np.multiply(prob.Vh.T, 1.0/(lam_l * prob.shs + lam_r)), np.dot(prob.Vh, rhs))
    elif prob.solve_method == 'solve':
        lhs = (lam_l * prob.AA + lam_r * np.identity(prob.A.shape[1]))
        u = sp.linalg.solve(lhs,rhs,sym_pos = True, assume_a = 'pos')
    else:
        raise NameError('py_ridge Recieved invalid solve_method ' + prob.solve_method)
    return u

def py_ridge_opt(prob,v,lam_l,lam_r, u_hat =None, Av = None, u0 = None, maxiter = 1000, tol = 1e-10):
    """
    """
    if Av is None:
        Av = np.dot(prob.A.T,v)
    rhs = lam_l * Av
    AAu_hat = np.dot(prob.Aast,np.dot(prob.A,u_hat))
    for i in range(maxiter):
        lam_r_prev = lam_r
        if (prob.solve_method == "eigh"):
            u_basis = np.dot(prob.eigs.T.conj(), rhs) 
            de = - np.sum(np.multiply(np.dot(np.multiply(prob.eigs, 1.0/((lam_l * prob.eigvals + lam_r)**3)),
                u_basis),rhs - AAu_hat - lam_r * u_hat))
            dde = np.sum(np.multiply(np.dot(np.multiply(prob.eigs, 1.0/((lam_l * prob.eigvals + lam_r)**4)),
                u_basis),3*rhs - 2*AAu_hat - 2*lam_r * u_hat))
            if (np.abs(de) < tol):
                print("Small derivative", np.abs(de))
                break
        elif (prob.solve_method == "svd"):
            u_basis = np.dot(prob.Vh, rhs) 
            de = - np.sum(np.multiply(np.dot(np.multiply(prob.Vh.T, 1.0/((lam_l * prob.shs + lam_r)**3)),
                u_basis),rhs - AAu_hat - lam_r * u_hat))
            dde = np.sum(np.multiply(np.dot(np.multiply(prob.Vh.T, 1.0/((lam_l * prob.shs + lam_r)**4)),
                u_basis),3*rhs - 2*AAu_hat - 2*lam_r * u_hat))

        p = -de/dde
        if (p*de > 0):
            p = -p
        lam_r = lam_r_prev + p
        alpha = 0.8
        while (lam_r < 0):
           lam_r = lam_r_prev + alpha* p
           alpha *=0.8
        if (np.abs(lam_r) < tol):
            print("No regularization", lam_r)
            break
    if (prob.solve_method == "eigh"):
        u = np.dot(np.multiply(prob.eigs, 1.0/(lam_l * prob.eigvals + lam_r)), u_basis)
    elif (prob.solve_method == "svd"):
        u = np.dot(np.multiply(prob.Vh.T, 1.0/(lam_l * prob.shs + lam_r)), u_basis)
    return u, 1.0/(1.0 + lam_r)


def py_lasso_pdhg(prob, v, lam_l, lam_r, 
    tau = 1.0, sigma = 1.0, Av = None,
    param_method = None, noise_std = None, u_hat = None,
    tol = 1e-16, maxiter = 1000, u0 = None, verbose = 0):
    """
    Solver for the so-called lasso problem min_u lam_l \|Au - v\|^2 + lam_r \|u\|_1
    using the primal dual hybrid gradient method.
    A, u and v are more commonly denoted as X, beta and Y in regression litterature.
    Parameters  :
        A : (N,M) ndarray. Contains the linear operator
            Note: Currently does not more general linear operators
        v : (N) ndarray
        lam_l : scalar. "Left" regularization parameter
        lam_r : scalar. "Right" regularization parameter
        method: string. 'chol' or 'eigh', describing what method to use for
            solving the problem
        eigvals: (N)  
    """
    if lam_l == 0:
        return np.zeros(prob.A.shape[1])
    elif lam_r == 0:
        return py_ridge(prob,v,lam_l,lam_r,Av = Av)
    elif lam_l < 0 or lam_r < 0 or (lam_l == 0 and lam_r == 0):
        raise RuntimeError('py_lasso_pdhg Requires at least one positive regularization parameter lam_l and lam_r')

    if Av is None:
        Av = np.dot((prob.A).T,  v) 

    if u0 is None:
        u = np.copy(Av)
        u_bar = np.copy(Av)
    else:
        u = np.copy(u0)
        u_bar = np.copy(u0)

    # Initial value for dual variable p 
    p = sigma * u
    p = lam_r * p/np.maximum(lam_r, np.abs(p))
    
    if prob.solve_method == 'chol':
        lhs = tau * lam_l * prob.AA + np.identity(prob.A.shape[1])
        chol = sp.linalg.cho_factor(lhs)
    elif prob.solve_method == 'solve':
        lhs = tau * lam_l * prob.AA + np.identity(prob.A.shape[1])
    
    # Extrapolation
    theta = 1.0

    lams = np.zeros(maxiter)
    duality_list = np.zeros(maxiter)
    res_list = np.zeros(maxiter)
    nonzero_list = np.zeros(maxiter)

    if param_method == "dp":
        if prob.solve_method != 'svd':
            raise RuntimeError('py_lasso_pdhg with discrepancy principle requires solve_method = "svd"')

    elif param_method == "opt":
        if prob.solve_method != 'svd' and prob.solve_method != "eigh":
            raise RuntimeError('py_lasso_pdhg with opt requires solve_method = "svd" or "eigh"')
        AAu_hat = np.dot(prob.AA,u_hat)
        AAAAu_hat = np.dot(prob.AA,AAu_hat)
        AAAv = np.dot(prob.AA,Av)
    elif param_method == "opt_alt":
        if prob.solve_method != 'eigh':
            raise RuntimeError('py_lasso_pdhg with opt_alt requires solve_method = "eigh"')
        vls = np.dot(np.multiply(prob.eigs, 1.0/(prob.eigvals)), np.dot(prob.eigs.T.conj(), Av))


    for i in range(maxiter):
        u_prev = np.copy(u)
        p_hat = p + sigma*u_bar
        p = lam_r * p_hat/np.maximum(lam_r, np.abs(p_hat))
        if param_method == "dp":
            lam_l_prev = lam_l
            vec = np.dot(prob.A, u - tau * p) -  v
            func = np.linalg.norm(np.dot(np.multiply(prob.U, 1.0/(tau * lam_l * prob.ssh + 1.0)), 
                np.dot(prob.U.T, vec)))**2 - np.prod(v.shape)*noise_std**2
            dfunc = - 2.0 * np.dot(np.dot(np.multiply(prob.U, tau * prob.ssh/(tau * lam_l * prob.ssh + 1.0)**3),
                 np.dot(prob.U.T, vec)), vec)
            d = -func/dfunc
            lam_l = lam_l_prev + 0.8*d
            alpha = 0.8
            while(lam_l < 0.0):
                lam_l = lam_l_prev + alpha * d
                alpha*=0.8

                    
        elif param_method == "opt" and i%50 == 0:
            lam_l_prev = lam_l
            u_tilde = u - tau * p
            AAu_tilde = np.dot(prob.AA,u_tilde)
        #    if (prob.solve_method == "eigh"):
        #        Av_basis = np.dot(prob.eigs.T.conj(),u_tilde + lam_l * tau * Av)
        #        de = -tau * np.sum(np.multiply(np.dot(np.multiply(prob.eigs, prob.eigvals/((tau * lam_l * prob.eigvals + 1.0)**3)),
        #            vec_basis),u_tilde + tau * lam_l * Av - tau*lam_l * AAu_hat - u_hat))
        #        dde =  -tau * tau * np.sum(np.multiply(np.dot(np.multiply(prob.eigs, prob.eigvals/((tau*lam_l * prob.eigvals + 1.0)**4)),
        #            vec_basis),-2 * (tau*lam_l * AAAAu_hat + AAu_hat) + 3 * AAu_tilde + 2 * tau * lam_l * AAAv - Av))
            if(prob.solve_method == "svd"):
                Av_basis = np.dot(prob.Vh,Av)
                uhat_basis = np.dot(prob.Vh,u_hat)
                denom2 = (tau*lam_l * prob.shs + 1.0)**2
                denom3 = (tau*lam_l * prob.shs + 1.0)*denom2
                denom4 = (tau*lam_l * prob.shs + 1.0)*denom3
                de1 = np.dot(np.multiply(prob.Vh.T, 1.0/denom2),Av_basis)
                dde1 = np.dot(np.multiply(prob.Vh.T, 1.0/denom3),Av_basis)
                de2 = np.dot(np.multiply(prob.Vh.T, prob.shs/denom3),uhat_basis) + tau * lam_l * np.dot(prob.AA,dde1)
                dde2 = np.dot(np.multiply(prob.Vh.T, prob.shs/denom4),uhat_basis + tau * lam_l * Av_basis)
                de = tau * np.sum(np.multiply(de1 - de2,u_tilde + tau * lam_l * Av - (tau*lam_l * AAu_hat + u_hat)))
                dde = -tau * tau * np.sum(np.multiply(dde1 - dde2,3 * AAu_tilde + 2 * tau * lam_l * AAAv - 2*(tau*lam_l * AAAAu_hat + AAu_hat) - Av))
            d = -de/dde
            if(d * de > 0):
                d = -d
            lam_l = lam_l_prev + d
            alpha = 0.5
            while(lam_l < 0.0):
                lam_l = lam_l_prev + alpha * d
                alpha*=0.5
        elif param_method == "opt_alt":
            AAp = np.dot(np.multiply(prob.eigs, 1.0/(prob.eigvals)), np.dot(prob.eigs.T, p))
            lam_l = np.linalg.norm(AAp)**2/np.sum(np.multiply(AAp,vls - u_hat))

        
        lams[i] = lam_l
        if param_method != "opt_alt":
            rhs = u - tau * p + tau * lam_l * Av
            if prob.solve_method == 'eigh':
                u = np.dot(np.multiply(prob.eigs, 1.0/(tau * lam_l * prob.eigvals + 1.0)), np.dot(prob.eigs.T, rhs))
            elif prob.solve_method == 'chol':
                u = sp.linalg.cho_solve(chol,rhs)
            elif prob.solve_method == 'solve':
                u = sp.linalg.solve(lhs,rhs)
            elif prob.solve_method == 'svd':
                u = np.dot(np.multiply(prob.Vh.T, 1.0/(tau * lam_l * prob.shs + 1.0)), np.dot(prob.Vh, rhs))
        elif param_method == "opt_alt" and lam_l > 0.0:
            u = vls - 1/lam_l * AAp
        else:
            u = vls
        res = np.linalg.norm(u - u_prev) 

        if (verbose):
            print("Iteration: ", i + 1, " |u - u_prev|: ", res, ", lam_l: ", lam_l)
        #if res < tol:
        #    break
        #p_hat = p + sigma*u_hat
        #p = lam_r * p_hat/np.maximum(lam_r, np.abs(p_hat))

        #duality_list[i] = np.linalg.norm(p + lam_l * np.dot(prob.A.T,np.dot(prob.A, u) - v))
        res_list[i] = res
        #nonzero_list[i] = len(u[np.nonzero(py_lasso_pg(prob,v,lam_l,lam_r, maxiter = 1, u0 = u, Av = Av))])
        #if (i%10 == 0):
        #    u = py_lasso_pg(prob,v,lam_l,lam_r, maxiter = 1, u0 = u, Av = Av)

        u_bar = u + theta*(u - u_prev)
    if verbose == 1: 
        plt.semilogy(lams[np.nonzero(lams)], label = "Regularization parameter")
        plt.semilogy(duality_list[np.nonzero(duality_list)], label = "Duality Gap")
        plt.semilogy(res_list[np.nonzero(res_list)], label = "|u_next - u|")
        plt.semilogy(nonzero_list[np.nonzero(nonzero_list)], label = "Nonzero u")
        plt.title(param_method)
        plt.xlabel("Iterations")
        plt.ylabel("Regularization parameter")
        plt.legend()
        plt.show()


    # Do a couple of projected gradient steps to properly round off
    u = py_lasso_pg(prob,v,lam_l,lam_r, maxiter = 1, u0 = u, Av = Av)
    if (param_method == "opt"):
        return u, lam_l/(1.0 + lam_l)
    return u

def py_lasso_admm(prob, v, lam_l, lam_r, 
    tau = 1.0, Av = None,
    param_method = None, noise_std = None, u_hat = None,
    tol = 1e-16, maxiter = 1000, u0 = None, verbose = 0):
    if lam_l == 0:
        return np.zeros(prob.A.shape[1])
    elif lam_r == 0:
        return py_ridge(prob,v,lam_l,lam_r,Av = Av)
    elif lam_l < 0 or lam_r < 0 or (lam_l == 0 and lam_r == 0):
        raise RuntimeError('py_lasso_pdhg Requires at least one positive regularization parameter lam_l and lam_r')

    if Av is None:
        Av = np.dot((prob.A).T,  v) 

    if u0 is None:
        u = np.copy(Av)
        z = np.copy(Av)
    else:
        u = np.copy(u0)
        z = np.copy(u0)
    
    # Lagrange parameter 
    s = u - z

    res_list = np.zeros(maxiter)
    lams = np.zeros(maxiter)
    if prob.solve_method == 'chol':
        lhs = lam_l * prob.AA + tau * np.identity(prob.A.shape[1])
        chol = sp.linalg.cho_factor(lhs)
    elif prob.solve_method == 'solve':
        lhs = lam_l * prob.AA + tau * np.identity(prob.A.shape[1])
    for i in range(maxiter):
        u_prev = np.copy(u)
        rhs = lam_l *Av + tau * z - s 
        if prob.solve_method == 'eigh':
            u = np.dot(np.multiply(prob.eigs, 1.0/(lam_l * prob.eigvals + tau)), np.dot(prob.eigs.T, rhs))
        elif prob.solve_method == 'chol':
            u = sp.linalg.cho_solve(chol,rhs)
        elif prob.solve_method == 'solve':
            u = sp.linalg.solve(lhs,rhs)
        elif prob.solve_method == 'svd':
            u = np.dot(np.multiply(prob.Vh.T, 1.0/(lam_l * prob.shs + tau)), np.dot(prob.Vh, rhs))
        temp = u + s/tau
        if param_method == "opt" and i%1 == 0:
            param_func = lambda lam: np.linalg.norm(prob.A.dot(np.sign(temp) * np.maximum(np.abs(temp) - lam/tau,0) - u_hat))**2
            #lam_r = sp.optimize.minimize_scalar(param_func, bounds = [0,2*lam_r],method = "bounded").x
            #lam_r = sp.optimize.minimize_scalar(param_func, bracket = [0.0, 2*lam_r], method = "golden",tol = 1e-10).x
            lam_r = sp.optimize.minimize_scalar(param_func, method = "brent",tol = 1e-10).x
        lams[i] = lam_r
        z = np.sign(temp) * np.maximum(np.abs(temp) - lam_r/tau, 0)
        s = s + tau * (u - z)
        res = np.linalg.norm(u - u_prev)
        res_list[i] = res
        if (verbose):
            print("Iteration: ", i + 1, " |u - u_prev|: ", res)
        if res < tol:
            break
    #z = py_lasso_pg(prob,v,lam_l,lam_r, maxiter = 1, u0 = z, Av = Av)
    if verbose:
        plt.semilogy(res_list)
        plt.semilogy(lams)
        plt.show()
    return z

def py_lasso_pg(prob,v,lam_l,lam_r, tol = 1e-8, maxiter = 1000, u0 = None, Av = None, verbose = False):
    """
    (Stochastic) Projected gradient method for solving 
    min lam_l/2 \|Au - v\| + \lam_r \|u\|_1
    """
    if Av is None:
        Av = np.dot(prob.A.T, v) 

    if u0 is None:
        u = np.copy(Av)
    else:
        u = np.copy(u0)
 

    # Step length
    if prob.solve_method == "eigh":
        tau = 1.0/(2*lam_l * np.max(prob.eigvals))
    elif prob.solve_method == "svd":
        tau = 1.0/(2*lam_l * np.max(prob.ssh))
    else:
        tau = 1.0/(lam_l * 10**5)

    duality_list = np.zeros(maxiter) 
    nonzero_list = np.zeros(maxiter)
    for i in range(maxiter):
        u_prev = np.copy(u)
        nablaL = lam_l * (np.dot(prob.A.T, np.dot(prob.A,u)) - Av)

        # Not actually p, p is +-nablaL
        p = u - tau*nablaL
        for j in range(u.shape[0]):
            if p[j] > tau * lam_r:
                u[j] = p[j] - tau * lam_r
            elif p[j] < - tau * lam_r:
                u[j] = p[j] + tau * lam_r
            else:
                u[j] = 0
        res = np.linalg.norm(u - u_prev) 

        duality_list[i] = np.linalg.norm(-nablaL + lam_l * np.dot(prob.A.T,np.dot(prob.A, u) - v))
        nonzero_list[i] = len(u[np.nonzero(u)])
        if (verbose):
            print("Iteration: ", i + 1, " |u - u_prev|: ", res)
        if res < tol:
            break
    if verbose == 1:
        plt.semilogy(duality_list[np.nonzero(duality_list)], label = "Duality Gap")
        plt.semilogy(nonzero_list[np.nonzero(nonzero_list)], label = "Nonzero u")
        plt.xlabel("Iterations")
        plt.ylabel("Regularization parameter")
        plt.legend()
        plt.show()

    return u 


def py_quadratic_denoise(prob,v, lam_l, lam_r, precond = True, u0 = None, Av = None,tol = 1.0e-4, maxiter=100 , verbose = False):
    """
    Preconditioned conjugate gradient solver for the problem
    min_u lam_l/2 \|u - f\|^2 + lam_r/2 \||D u |\|^2,
    which has the "explicit" solution
    u = (I + D^ast D)^{-1}v
    """
    u = np.copy(v)
    Au = lam_l * u + lam_r * prob.Dast(prob.D(u))
    r = lam_l * v - Au

    # Setup preconditioner, fker = F(I + D^/ast D)
    if precond:
        norm = None
        fshape = [sp.fft.next_fast_len(v.shape[0] + 10, True), 
               sp.fft.next_fast_len(v.shape[1] + 10, True)] 
        workers = 8
        
        # Setup kernels and apply ffts
        kerid = np.array([[0,0,0],[0,1,0],[0,0,0]])
        kerlap = np.array([[0, -1.0, 0], [-1.0, 4.0, -1.0],[0,-1.0,0]]) 
        fkerid = sp.fft.rfft2(kerid, s = fshape, norm = norm,workers = workers)
        fkerlap = sp.fft.rfft2(kerlap, s = fshape, norm = norm,workers = workers)
        
        fr = sp.fft.rfft2(np.pad(r, ((1, fshape[0] - r.shape[0] - 1),(1, fshape[1] - r.shape[1] - 1))), s = fshape, norm = norm, workers = workers)
        z = sp.fft.irfft2(fr/(lam_l * fkerid + lam_r * fkerlap), norm = norm, workers = workers)[:r.shape[0], :r.shape[1]]
        p = z
        rz_next = np.sum(np.multiply(r,z))
    else:
        p = r

    rr_next = np.linalg.norm(r,'fro')**2
    rr0 = rr_next

    for i in range(maxiter):
        if (verbose):
            print("Iteration: ", i , " res^2: ", rr_next)
        Ap = lam_l * p + lam_r * prob.Dast(prob.D(p))
        if precond:
            rz_curr = rz_next
            alpha = rz_curr/np.sum(np.multiply(p,Ap))
        else:
            rr_curr = rr_next
            alpha = rr_curr/np.sum(np.multiply(p,Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        if precond:
            fr = sp.fft.rfftn(np.pad(r, ((1, fshape[0] - r.shape[0] - 1),(1, fshape[1] - r.shape[1] - 1))) , s = fshape, norm = norm, workers = workers)
            z = sp.fft.irfftn(fr/(lam_l * fkerid + lam_r * fkerlap), norm = norm, workers = workers)[:r.shape[0], :r.shape[1]]
            rz_next = np.sum(np.multiply(r,z))
            rr_next = np.linalg.norm(r,'fro')**2
            if rr_next/rr0 < tol:
                break
            beta = rz_next/rz_curr
            p = z + beta * p
        else:
            rr_next = np.linalg.norm(r,'fro')**2
            if rr_next/rr0 < tol:
                break
            beta = rr_next/rr_curr
            p = r + beta * p
    if verbose:
        print("Final residual: ", rr_next)
    return u

def py_quadratic_SR(prob,v, lam_l, lam_r, precond = False, u0 = None, Av = None,tol = 1.0e-4, maxiter=100 , verbose = False):
    """
    Preconditioned conjugate gradient solver for the problem
    min_u lam_l/2 \|u - f\|^2 + lam_r/2 \||D u |\|^2,
    which has the "explicit" solution
    u = (lam_lI + lam_rD^ast D)^{-1}v
    """
    u = prob.Aast(v)
    if Av is None:
        Av = prob.Aast(v)
    Au = lam_l * prob.AA(u) + lam_r * prob.Dast(prob.D(u))
    r = lam_l * prob.Aast(v) - Au
    if precond:
        norm = None
        fshape = [sp.fft.next_fast_len(v.shape[0] + 10, True), 
               sp.fft.next_fast_len(v.shape[1] + 10, True)] 
        workers = 8
        
        # Setup kernels and apply ffts
        kerid = np.array([[0,0,0],[0,1,0],[0,0,0]])
        kerlap = np.array([[0, -1.0, 0], [-1.0, 4.0, -1.0],[0,-1.0,0]]) 
        fkerid = sp.fft.rfft2(kerid, s = fshape, norm = norm,workers = workers)
        fkerlap = sp.fft.rfft2(kerlap, s = fshape, norm = norm,workers = workers)
        
        fr = sp.fft.rfft2(np.pad(r, ((1, fshape[0] - r.shape[0] - 1),(1, fshape[1] - r.shape[1] - 1))), s = fshape, norm = norm, workers = workers)
        z = sp.fft.irfft2(fr/(lam_l * fkerid + lam_r * fkerlap), norm = norm, workers = workers)[:r.shape[0], :r.shape[1]]
        p = z
        rz_next = np.sum(np.multiply(r,z))
    else:
        p = r
    # Setup preconditioner, fker = F(I + D^/ast D)

    rr_next = np.linalg.norm(r,'fro')**2
    rr0 = rr_next

    for i in range(maxiter):
        if (verbose):
            print("Iteration: ", i , " res^2: ", rr_next)
        Ap = lam_l * prob.AA(p) + lam_r * prob.Dast(prob.D(p))
        if precond:
            rz_curr = rz_next
            alpha = rz_curr/np.sum(np.multiply(p,Ap))
        else:
            rr_curr = rr_next
            alpha = rr_curr/np.sum(np.multiply(p,Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        if precond:
            fr = sp.fft.rfftn(np.pad(r, ((1, fshape[0] - r.shape[0] - 1),(1, fshape[1] - r.shape[1] - 1))) , s = fshape, norm = norm, workers = workers)
            z = sp.fft.irfftn(fr/(lam_l * fkerid + lam_r * fkerlap), norm = norm, workers = workers)[:r.shape[0], :r.shape[1]]
            rz_next = np.sum(np.multiply(r,z))
            rr_next = np.linalg.norm(r,'fro')**2
            if rr_next/rr0 < tol:
                break
            beta = rz_next/rz_curr
            p = z + beta * p
        else:
            rr_next = np.linalg.norm(r,'fro')**2
            if rr_next/rr0 < tol:
                break
            beta = rr_next/rr_curr
            p = r + beta * p
    if verbose:
        print("Final residual: ", rr_next)
    return u

def py_pdps_denoise(prob, v, lam_l, lam_r, param_method = None,
    noise_std = None, u_hat = None, Av = None,
    u0 = None, tau = 0.3, sig = 0.3, extrapolate = True, theta = 1.0, 
    tol = 1.0e-4, maxiter = 100, verbose = 0):
    """
    Primal-dual proximal splitting (Chambolle-Pock) for solving denoising problem
    min 0.5lam_l \|u - v\|^2 + lam_r TV(u)

    The step lengths to satisfy
    sig tau < 1/\|D\|^2,
    with \|D\|^2 \approx 8 for most difference operators 

    """
    if u0 is None:
        u = np.zeros(v.shape, v.dtype)
        u_prev = np.zeros(v.shape, v.dtype) 
        u_next = np.zeros(v.shape, v.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_next = np.copy(u0)
    p = sig*prob.D(v)
    Dp = prob.Dast(p)

    if param_method == "dp":
        cc = np.prod(v.shape) * noise_std**2
    elif param_method == "opt":
        vv = np.linalg.norm(v,'fro')**2
        vu_hat = np.sum(np.multiply(v,u_hat))

    lams = np.zeros(maxiter)
    res_list = np.zeros(maxiter)    

    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        if extrapolate:
            p = p + sig * prob.D(u_next)
        else:
            p = p + sig * prob.D(u)
        p = lam_r * p / np.maximum(lam_r,norm(p))
        Dp = prob.Dast(p)

        if param_method == "dp":
            # Only for denoising
            lam_l_prev = lam_l
            #temp = np.linalg.norm(u - tau * Dp - v - tau*(u - u_hat),'fro')**2
            temp = np.linalg.norm(u - tau * Dp - v,'fro')**2
            e = 1/(1 + lam_l * tau)**2 * temp - cc
            de = - 2*tau/(1 + lam_l * tau)**3 * temp
            lam_l = lam_l_prev - e/de
            if (lam_l < 0.0):
                alpha = 0.5
                for m in range(50):
                    lam_l = lam_l_prev - alpha * e/de
                    if 0.0 < lam_l:
                        break
                    alpha*=0.5

        elif param_method == "UPRE":
            # Should remove this, for next article!
            lam_l_prev = lam_l
            temp = np.linalg.norm(u - tau * Dp - v, 'fro')**2
            de = -2 * tau/(1 + lam_l * tau)**3 * temp + 2 * tau * noise_std**2 * np.prod(v.shape) * ( 1/(1 + lam_l * tau) - lam_l * tau/(1 + lam_l * tau)**2) 
            dde = 6 * tau * tau/(1 + lam_l * tau)**4 * temp + 2 * tau * noise_std**2 * np.prod(v.shape) * (-tau * (1 + lam_l)/(1 + lam_l * tau)**2 + 2 * tau * tau/(1 + lam_l * tau)**3)
            lam_l = lam_l_prev - de/dde
            if (de * de/dde > 0):
                de = -de

            if (lam_l < 0.0):
                alpha = 0.5
                for m in range(50):
                    lam_l = lam_l_prev - alpha * de/dde
                    if 0.0 < lam_l:
                        break
                    alpha*=0.5
        elif param_method == "opt":
            u_tilde = u - tau * Dp
            vu_tilde = np.sum(np.multiply(v,u_tilde))
            u_hatu_tilde = np.sum(np.multiply(u_hat,u_tilde))
            u_tildeu_tilde = np.linalg.norm(u_tilde,'fro')**2
            lam_l = 1.0/tau * (- vu_tilde + u_tildeu_tilde - u_hatu_tilde +  vu_hat)/(vv - vu_tilde + u_hatu_tilde - vu_hat) # Is the sign here incorrect?
        lams[i] = lam_l   
        if param_method == "dp":
            u = 1.0/(1.0 + lam_l*tau) * (u - tau * Dp + tau * lam_l * v - tau*(u - u_hat)) 
        else:
            u = 1.0/(1.0 + lam_l*tau) * (u - tau * Dp + tau * lam_l * v) 
        if extrapolate:
            u_next = u + theta*(u - u_prev)
        res_list[i] = np.linalg.norm(u - u_prev)
        #if (res_list[i] < tol):
        #    break
        if verbose:
            print("Iteration: ", str(i), ", |u - u_prev| = ", res_list[i], "lam_l = ", lam_l)
    if verbose == 1: 
        plt.semilogy(lams[np.nonzero(lams)])
        plt.semilogy(res_list[np.nonzero(res_list)])
        plt.title(param_method)
        plt.xlabel("Iterations")
        plt.ylabel("Regularization parameter")
        plt.show()
    return u

def py_pdps_SR(prob, v, lam_l, lam_r, param_method = None,
    noise_std = None, u_hat = None, Av = None,
    u0 = None, tau = 0.33, sig = 0.33, theta = 1.0, 
    tol = 1.0e-4, maxiter = 100, verbose = 0):
    """
    Primal-dual proximal splitting (Chambolle-Pock) for solving SR problem
    min 0.5lam_l \|Au - v\|^2 + lam_r TV(u)
    where A is a downsampling operator with A A^\ast = cI for some cosntant c

    The step lengths to satisfy
    sig tau < 1/\|D\|^2,
    with \|D\|^2 \approx 8 for most difference operators 

    """
    if Av is None:
        Av = prob.Aast(v)
    if u0 is None:
        u = np.copy(Av)
        u_prev = np.copy(Av) 
        u_next = np.copy(Av) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
        u_next = np.copy(u0)

    
    p = sig*prob.D(u)
    Dp = prob.Dast(p)

    if param_method == "projopt":
        vv = 0.25 * np.linalg.norm(Av,'fro')**2
        vu_hat = 0.25 * np.sum(np.multiply(Av,u_hat))
    elif param_method == "dp":
        cc = np.prod(v.shape) * noise_std**2

    lams = np.zeros(maxiter)
    res_list = np.zeros(maxiter)    

    for i in range(maxiter): # and min > tol)):
        u_prev = np.copy(u)
        p = p + sig * prob.D(u_next)
        p = lam_r * p / np.maximum(lam_r,norm(p))
        Dp = prob.Dast(p)
        if param_method == "dp":
            lam_l_prev = lam_l
            temp = np.linalg.norm(prob.A(u - tau * Dp) - v,'fro')**2
            e = 1/(1 + 0.25 * lam_l * tau)**2 * temp - cc
            de = - 2* 0.25 * tau/(1 + 0.25 * lam_l * tau)**3 * temp
            lam_l = lam_l_prev - e/de
            if (lam_l < 0.0):
                alpha = 0.5
                for m in range(50):
                    lam_l = lam_l_prev - alpha * e/de
                    if 0.0 < lam_l:
                        break
                    alpha*=0.5 
        if param_method == "projopt":
            u_tilde = prob.AA(u - tau * Dp)
            vu_tilde =  np.sum(np.multiply(Av,u_tilde))
            u_hatu_tilde = np.sum(np.multiply(u_hat,u_tilde))
            u_tildeu_tilde = 4 * np.linalg.norm(u_tilde,'fro')**2
            lam_l = 1.0/tau * (- vu_tilde + u_tildeu_tilde - u_hatu_tilde +  vu_hat)/(vv - vu_tilde + u_hatu_tilde - vu_hat)
        lams[i] = lam_l 
        if lam_l < 0.0:
            u = u - tau * Dp
        else:
            rhs = u - tau * Dp + tau * lam_l * Av
            if prob.solve_method == "cg":
                Au = tau * lam_l * prob.AA(u) + u
                r = rhs - Au
                z = r
                rr_next = np.linalg.norm(r,'fro')**2
                rr0 = rr_next
                if(rr_next == 0.0):
                    u = rhs
                else:
                    for i in range(4):
                        if (verbose):
                            print("Iteration: ", i , " res^2: ", rr_next)
                        Az = tau * lam_l * prob.AA(z) + z
                        rr_curr = rr_next
                        alpha = rr_curr/np.sum(np.multiply(z,Az))
                        u = u + alpha * z
                        r = r - alpha * Az
                        rr_next = np.linalg.norm(r,'fro')**2
                        if rr_next/rr0 < 1e-10:
                            break
                        beta = rr_next/rr_curr
                        z = r + beta * z
        u_next = u + theta*(u - u_prev)
        res_list[i] = np.linalg.norm(u - u_prev)
        if (res_list[i] < tol):
            break
        if verbose:
            print("Iteration: ", str(i), ", |u - u_prev| = ", res_list[i])
    if verbose == 1: 
        plt.semilogy(lams[np.nonzero(lams)])
        plt.semilogy(res_list[np.nonzero(res_list)])
        plt.title(param_method)
        plt.xlabel("Iterations")
        plt.ylabel("Regularization parameter")
        plt.show()
    return u

def py_pg_denoise(prob, v, lam_l, lam_r, param_method = None,
    noise_std = None, u_hat = None, Av = None,
    u0 = None, tau = 0.25, extrapolate = True, theta = 1.0, 
    tol = 1.0e-4, maxiter = 100, verbose = 0):
    """
    Primal-dual proximal splitting (Chambolle-Pock) for solving denoising problem
    min 0.5lam_l \|u - v\|^2 + lam_r TV(u)

    The step lengths to satisfy
    sig tau < 1/\|D\|^2,
    with \|D\|^2 \approx 8 for most difference operators 

    """
    if u0 is None:
        u = np.zeros(v.shape, v.dtype)
        u_prev = np.zeros(v.shape, v.dtype) 
    else:
        u = np.copy(u0)
        u_prev = np.copy(u0)
    p = tau*prob.D(v)
    Dp = prob.Dast(p)
    lams = np.ones(maxiter+1)*lam_l
    resplist = np.zeros(maxiter)
    resulist = np.zeros(maxiter)
    derivlist = np.zeros(maxiter)
    derivlist2 = np.zeros(maxiter)
    res = np.linalg.norm(v)
    for i in range(maxiter):
        u_prev = np.copy(u)
        p_prev = np.copy(p)
        p = p + tau*prob.D(u)
        p = lam_r * p / np.maximum(lam_r, norm(p)) 
        Dp = prob.Dast(p)
        if param_method == 'opt':
            normdivp =  np.linalg.norm(Dp,'fro')**2
            lams[i+1] = normdivp/(np.sum(np.multiply(Dp,v - u_hat)))
        
        u = v - (1/lams[i+1])*Dp 
        tau = 0.99*(2.0*lams[i+1]/8.0)
        if np.linalg.norm(u - u_prev)/res< tol:
            break
        #p = p + tau*prob.D(u)
        #p = lam_r * p / np.maximum(lam_r, norm(p))  
        if verbose:
            resulist[i] = np.linalg.norm(u - u_prev)
            resplist[i] = np.linalg.norm(p - p_prev)
        if i > 1:
            #Dp = prob.Dast(p)
            dpdlam = (p - p_prev)/(lams[i] - lams[i-1])
            derivlist[i] = np.abs(np.sum(np.multiply(- 1/lams[i]**2 * Dp, v - 1/lams[i] * Dp - u_hat)) + np.sum(np.multiply(1/lams[i] * prob.Dast(dpdlam),v - 1/lams[i] * Dp - u_hat)))
            derivlist2[i] = np.abs(np.sum(np.multiply(Dp, v - 1/lams[i] * Dp - u_hat)))
    if verbose:
        plt.semilogy(resulist, label = "Primal")
        plt.semilogy(resplist, label = "Dual")
        plt.semilogy(lams, label = "Lambda")
        plt.semilogy(derivlist)    
        plt.semilogy(derivlist2)    

        plt.legend()
        plt.show()
    return u



def f2py_cp_denoise(f, lam, u0 = None, tau = 0.25, sig = 0.25, theta = 1.0, acc = False, tol = 1.0e-10):
    n = f.shape[0]
    if u0 is None:
        res = np.zeros(f.shape, dtype = np.float32, order = 'F')
    else:
        res = u0
    chambollepock_denoise(f,res, lam, tau, sig, theta, acc, tol, n)
    return res

    
