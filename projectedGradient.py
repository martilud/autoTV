from utils import *

def projected_gradient(f, lam, tau = 0.25, tol=1.0e-10):
    """
    Uses Chambolle's algorithm from 2004
    Note lam should be given as the inverse of lambda
    """
    lam = 1/lam
    p = np.zeros((2,) + f.shape, f.dtype)
    new_p = np.zeros((2,) + f.shape, f.dtype)
    u = np.copy(f)
    divp = np.zeros(f.shape, f.dtype)
    diff = np.inf
    f_lam = f/lam
    i = 0
    maxiter = 500
    dualitygap_list = np.zeros(maxiter+1)
    dualitygap_list[0] = dualitygap_denoise(lam,u,divp,f)

    while (diff > tol and i < maxiter):
        grad_div_p_i = grad(divp - f_lam)
        grad_div_p_i = grad(f - lam*divp)
        norm1_grad_div_p_i = better_norm1(grad_div_p_i)
        new_p = (p +  tau * grad_div_p_i)/(1 + tau * norm1_grad_div_p_i)
        diff = np.max(norm1(new_p - p))
        p = new_p
        divp = div(p)
        #losslist[i] = loss(f - lam*divp, f, lam) 
        dualitygap_list[i+1] = dualitygap_denoise(lam,f - lam*divp,divp,f)
        i += 1
    print(i)
    return f - lam*divp, dualitygap_list

def projected_gradient_alt_conv(f, lam,  tau = 0.25, tol=1.0e-10, u0 = None):
    if u0 is  None:
        #u = np.copy(f)
        u = np.zeros(f.shape, f.dtype)
    else:
        u = np.copy(u0)
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = np.zeros(f.shape, f.dtype)
    maxiter = 500
    dualitygap_list = np.zeros(maxiter+1)
    dualitygap_list[0] = dualitygap_denoise(lam,u,divp,f)
    psnr_list = np.zeros(maxiter)
    i = 0
    res = tol + 1.0
    while (i < maxiter and res > tol):
        u_prev = np.copy(u)
        u = f + divp
        grad_div_p_i = grad(u)
        p_hat = p + tau*grad(u)
        p = lam * p_hat / np.maximum(lam, better_norm1(p_hat))
        divp = div(p)
        dualitygap_list[i+1] = dualitygap_denoise(lam,u,divp,f)
        res = dualitygap_list[i+1]/dualitygap_list[0]
        psnr_list[i] = psnr(u,u_prev)
        i += 1
    return u, dualitygap_list, psnr_list, i

def projected_gradient_alt(f, lam,  tau = 0.25, tol=1.0e-10, u0 = None):
    if u0 is  None:
        #u = np.copy(f)
        u = np.zeros(f.shape, f.dtype)
    else:
        u = np.copy(u0)
    p = np.zeros((2,) + f.shape, f.dtype)
    p_hat = np.zeros((2,) + f.shape, f.dtype)
    divp = np.zeros(f.shape, f.dtype)
    maxiter = 500
    dualitygap_init = dualitygap_denoise(lam,u,divp,f)
    i = 0
    res = tol + 1.0
    while (i < maxiter and res > tol):
        u = f + divp
        grad_div_p_i = grad(u)
        p_hat = p + tau*grad(u)
        p = lam * p_hat / np.maximum(lam, better_norm1(p_hat))
        divp = div(p)
        if (i%5 == 0):
            res = dualitygap_denoise(lam,u,divp,f)/dualitygap_init
        i += 1
    return u, dualitygap_list, psnr_list, i


def plotR(noisy,true):
    N = 100
    t_list = np.linspace(0.0,1.0, N)
    lam_list = (1.0-t_list[1:N-1])/t_list[1:N-1]
    R_list = np.zeros(N)
    for i in range(1,N-1):
        R_list[i] = R(t_list[i],noisy,true)
       #R_list[i] = more_psnr(t_list[i],noisy,true)
    t_opt = t_list[np.argmin(R_list[1:N-1])]
    print("optimal t", t_opt)
    print("corresponding lam", (1-t_opt)/t_opt)
    #plt.plot(lam_list,R_list[1:N-1])
    plt.plot(t_list[1:N-1],R_list[1:N-1])
    plt.xlabel('t')
    plt.ylabel('R(t)')
    plt.show()
    return t_opt

if __name__ == "__main__":
    Originalf = scipy.misc.imread('images/einstein.jpg', 'jpg')
    #Originalf = Originalf/255.0 
    sigma = 10
    f = add_gaussian_noise(Originalf, sigma)
    t_opt = plotR(f,Originalf)
    lam = (1-t_opt)/t_opt
    #lam = 23.75
    result1 = total_variation_2D(f,1/lam) 
    result2 = ptv.tv1_2d(f,lam,n_threads=4)
    
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].imshow(f, cmap = 'gray')
    ax[0].set_title("Noisy, sigma = " + str(sigma))
    ax[1].imshow(result1, cmap ='gray')
    ax[1].set_title("Solution")
    ax[2].imshow(result2, cmap = 'gray')
    ax[2].set_title("Solution")
    plt.show()
