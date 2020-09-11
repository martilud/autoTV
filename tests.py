from utils import *
from chambollepock import *
from quadraticTV import *
import time
from PIL import Image
import os
import numpy.random
from parameterselection import *
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
#from interpolation import *
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, rfftn, irfftn
from scipy.fftpack import dct, idct
from scipy.signal import fftconvolve, convolve, gaussian
import speed
from interpolation import *

def plotssimd():
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    N = 200
    t_list = np.linspace(0.01, 0.999, N)
    print(t_list[183])
    ssim_lists = np.zeros((4, N))
    win_sizes = [33, 17, 9]
    for i in range(0,N):
        t = t_list[i]
        u_t = f2py_cp_denoise(f, (1-t)/t)
        for j, win_size in enumerate(win_sizes):
            ssim_lists[j,i] = ssim(u_t, u, win_size = win_size)
    for i, win_size in enumerate(win_sizes):
        plt.plot(t_list, ssim_lists[i], label = 'Window size: '+ str(win_size) + r'$\times$' + str(win_size))
        print(np.argmax(ssim_lists[i]))
        plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel('SSIM')
    plt.grid()
    plt.show()
    
def plotgaussian():
    u = imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    m = u.shape[0]
    sig_noise = 0.1
    sigmas = [0.1]
    N = 100
    sigs = np.linspace(0,2,N)
    u_blur = np.zeros((N,m,m))
    psnrs = np.zeros(N)
    f = add_gaussian_noise(u, sigma = sig_noise)
    for sig in sigmas:
        f = add_gaussian_noise(u, sigma = sig)

        for i,sig_blur in enumerate(sigs):
            u_blur = scipy.ndimage.gaussian_filter(f,sig_blur)
            psnrs[i] = psnr(u,u_blur)
        print("OPTIMAL SIG BLUR:", sigs[np.argmax(psnrs)])
        opt_sig = sigs[np.argmax(psnrs)]
    #    plt.plot(sigs,psnrs)
    #plt.show()
    sig_blur = 1.2
    #t_list = np.linspace(0.0,1.0,N)[1:N-1]
    #u_blur = scipy.ndimage.gaussian_filter(f,opt_sig)
    #_, R_list = gridSearch(f,u,N, f2py_cp_denoise, plot = False)
    #_, R_list_hat = gridSearch(f,u_blur, N, f2py_cp_denoise, plot = False)
    #plt.plot(t_list,R_list, label = 'True loss')
    #plt.plot(t_list,R_list_hat, label = 'Approximate loss')
    #plt.xlabel(r'$t$')
    #plt.ylabel(r'$E(t)$')
    #plt.legend()
    #plt.grid()
    #plt.show()
    u_blur = scipy.ndimage.gaussian_filter(f,sig_blur)
    u_rec, t = optTV(f,u_blur, f2py_cp_denoise)
    u_opt, t_opt = optTV(f,u, f2py_cp_denoise)
    print(t, t_opt)
    print(psnr(u_rec,u), psnr(u_opt,u))

def testMean():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sigma = 0.1
    n = 1000
    image_array = np.zeros((n,)+u.shape)
    avg_image_array = np.zeros((n,)+u.shape)
    psnr_array = np.zeros(n)
    avg_img = np.zeros(u.shape)
    for i in range(n):
        image_array[i] = add_gaussian_noise(u,sigma)
        avg_image_array[i] =np.average(image_array[:(i+1)], axis= 0)
        psnr_array[i] = psnr(u,avg_image_array[i])
    plt.plot(psnr_array)
    plt.show()

def testDownsampled():
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)

    orig_shape = f.shape
    n = 4
    n_rec = 4
    nn = np.ceil(np.sqrt(n))

    sigma_blur = 0.3

    avg_img_recursive = createEstimator_recursive(f,n = n_rec, depth = 1, curr_depth =1, blur = True, sigma_blur = sigma_blur/4)
    image_array_random = createDownsampled_random(f,n, N = 10, blur = True, sigma_blur = sigma_blur)
    image_array_fixed = createDownsampled(f,n, blur = True, sigma_blur = sigma_blur)
    image_array_upsampled_random = createUpsampled(image_array_random,orig_shape)
    image_array_upsampled_fixed = createUpsampled(image_array_fixed,orig_shape)
    #fig = plt.figure(figsize=(nn,nn))
    #for i in range(n):
    #    fig.add_subplot(nn,nn,i+1)
    #    plt.imshow(image_array_random[i], cmap = "gray")
    #plt.show()

    avg_img_random = np.average(image_array_upsampled_random, axis=0)
    avg_img_fixed = np.average(image_array_upsampled_fixed, axis=0)

    num_plot = 4
    fig,ax = plt.subplots(1, num_plot)
    ax[0].imshow(f, cmap="gray")
    ax[0].title.set_text("NOISY")
    ax[1].imshow(avg_img_random, cmap = "gray")
    ax[1].title.set_text("RANDOM")
    ax[2].imshow(avg_img_fixed, cmap = "gray")
    ax[2].title.set_text("FIXED")
    ax[3].imshow(avg_img_recursive, cmap = "gray")
    ax[3].title.set_text("RECURSIVE")
    plt.show()

    print("SSIM orig:", ssim(f,u, data_range = 1.0))
    print("PSNR orig:", psnr(f,u))
    print("SSIM random:", ssim(f,avg_img_random, data_range = 1.0))
    print("PSNR random:", psnr(avg_img_random,u))
    print("SSIM fixed:", ssim(f,avg_img_fixed, data_range = 1.0))
    print("PSNR fixed:", psnr(avg_img_fixed,u))
    print("SSIM recursive:", ssim(u,avg_img_recursive, data_range = 1.0))
    print("PSNR recursive:", psnr(u,avg_img_recursive))

    #t_opt, R_list = gridSearch(f, u, 50, plot=False)
    #t_opt_hat, R_list_hat = gridSearch(f, avg_img, 50, plot=False)
    #plt.plot(R_list)
    #plt.plot(R_list_hat)
    #plt.show()
    #u_opt_DC, t_opt_DC = discrepancy_ruleTV(f, sigma, tau = 1.0, lam_init = 1.0, q = 0.9)
    tol = 1e-4
    u_opt, t_opt = optTV_golden(f,u, tol = tol)
    print("SSIM rec orig:", ssim(f,u_opt, data_range = 1.0))
    print("PSNR rec orig:",psnr(u_opt,u))

    u_opt_recursive, t_opt_recursive = optTV_golden(f, avg_img_recursive, tol = tol)
    print("SSIM rec recursive:", ssim(f,u_opt_recursive, data_range = 1.0))
    print("PSNR rec recursive:", psnr(u_opt_recursive,u))

    u_opt_random, t_opt_random = optTV_golden(f, avg_img_random, tol = tol)
    print("SSIM rec random:", ssim(f,u_opt_random, data_range = 1.0))
    print("PSNR rec random:", psnr(u_opt_random,u))

    u_opt_fixed, t_opt_fixed = optTV_golden(f, avg_img_fixed, tol = tol)
    print("SSIM rec fixed:", ssim(f,u_opt_fixed, data_range = 1.0))
    print("PSNR rec fixed", psnr(u_opt_fixed,u))

    print("t opt, random, fixed, recursive:", t_opt, t_opt_random, t_opt_fixed, t_opt_recursive)

    fig,ax = plt.subplots(1, num_plot)
    ax[0].imshow(u_opt, cmap="gray")
    ax[0].title.set_text("OPTIMAL")
    ax[1].imshow(u_opt_random, cmap = "gray")
    ax[1].title.set_text("RANDOM")
    ax[2].imshow(u_opt_fixed, cmap = "gray")
    ax[2].title.set_text("FIXED")
    ax[3].imshow(u_opt_recursive, cmap = "gray")
    ax[3].title.set_text("RECURSIVE")
    plt.show()

    #print(t_opt, t_opt_hat, t_opt_DC )
    #print(psnr(u_opt_DC,u))
    exit()
    """
    image_array_upsampled = createUpsampled(image_array,orig_shape)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n):
        fig.add_subplot(nn,nn,i+1)
        plt.imshow(image_array_upsampled[i], cmap = "gray")
    plt.show()
    #basis_img = modified_gram_schmidt_vec(image_array_upsampled)
    """

    t_opt_hat, R_list = gridSearch(f, avg_img, N = 30, plot=True) 
    basis_img = modified_gram_schmidt(avg_img - image_array_upsampled[:-1])

    #basis_img_img = basis_img.reshape((basis_img.shape[0],) +  orig_shape)
    fig = plt.figure(figsize=(nn,nn))
    for i in range(n-1):
        fig.add_subplot(nn,nn,i+1)
        #print(np.inner(basis_img[0], basis_img[i]))
        #print(np.linalg.norm(basis_img[i]))
        plt.imshow(basis_img[i], cmap = "gray")
        #plt.imshow(basis_img_img[i], cmap = "gray")
    plt.show()
    h = 8
    basis_img = basis_img[:h, :]
    N = 50
    #t_opt = 0.9228 #256 image
    #t_opt, R_list = gridSearch(f,u,N)
    t_opt_hat, t_list,  R_list_hat = gridSearch_basis(f, basis_img, avg_img, N)
    #print(t_opt, t_opt_hat)
    plt.plot(t_list, R_list_hat)
    plt.vlines(t_opt, ymin = np.min(R_list_hat), ymax = np.max(R_list_hat))
    #plt.plot(R_list)
    plt.show()

def testInterpolation():
    u = imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sig = 0.01
    f = add_gaussian_noise(u, sigma = sig)
    print(psnr(u,f))
    newshape = (256,256)
    shape = u.shape
    sigma_blur = 1.2
    image_array_fixed = createDownsampled(f,n=4, blur = True, sigma_blur = sigma_blur)
    image_array_upsampled_fixed = createUpsampled(image_array_fixed,shape)
    avg_img = np.average(image_array_upsampled_fixed, axis = 0)
    print(psnr(avg_img,u))
    #xx,yy = np.meshgrid(np.arange(u.shape[0]),np.arange(u.shape[1]))
    #print(xx.shape)
    #plt.scatter(xx,yy, c=u[::-1],s=0.01)
    #plt.show()
    order_down = 0
    order_up = 5
    aa = True
    aa_up = False
    aa_sig = 1.0
    sigmas = np.linspace(0,2,10)
    modes = ['constant', 'edge', 'symmetric','reflect','wrap']
    mode = 'symmetric'
    result_mat = np.zeros((6,6))
    max_psnr = 0.0
    for sig in sigmas:
        print(sig)
        gauss = resize(f, shape, order = 0, mode = mode, anti_aliasing = aa, anti_aliasing_sigma = sig)
        print(psnr(gauss,u))
        for orderd in range(6):
            for orderu in range(6):
                down = resize(f, newshape, order = orderd, mode=mode, anti_aliasing = aa,anti_aliasing_sigma = sig)
                up = resize(down, shape, order = orderu, mode=mode, anti_aliasing = aa_up)
                result_mat[orderd, orderu] = psnr(up,u)
                #print(orderd, orderu, psnr(up,u))
        if(np.max(result_mat)> max_psnr):
            max_psnr = np.max(result_mat)
            max_list = [max_psnr, np.where(result_mat == np.max(result_mat)), sig, mode]

        print(result_mat)
        print(np.max(result_mat))
        print(np.where(result_mat == np.max(result_mat)))
    print(max_list)
    #result1 = nearest(u, newshape)
    #image = Image.fromarray(u)
    #image = image.resize(newshape, Image.NEAREST)
    #result2 = np.array(image)
    #print(psnr(result1,result2))
    #plt.imshow(result1,cmap = "gray")
    #plt.show()

def test_gauss():
    u = imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sig = 0.1
    f = add_gaussian_noise(u, sigma = sig)
    #plt.imshow(f, cmap= 'gray')
    #plt.show()
    print("NOISY PSNR:",psnr(u,f))
    blur_sigs = np.linspace(0,3,100)
    blurs = np.zeros(100)
    i = 0
    for blur_sig in blur_sigs:
        blurred = scipy.ndimage.gaussian_filter(f,blur_sig)
        blurs[i] = psnr(blurred,u)
        i+=1
    best_blur_sig = blur_sigs[np.argmax(blurs)]
    blurred = scipy.ndimage.gaussian_filter(f,best_blur_sig)
    print("GAUSSIAN BLUR PSNR:",psnr(blurred,u))
    shape = f.shape
    image_array_fixed = createDownsampled(f,n=4, blur = True, sigma_blur = 0.0)
    image_array_upsampled_fixed = createUpsampled(image_array_fixed,shape)
    avg_img = np.average(image_array_upsampled_fixed, axis = 0)
    print("FIXED AVG (NO BLUR) PSNR:",psnr(avg_img,u))

    tol = 1e-4
    u_opt, t_opt = optTV_golden(f,u, tol = tol)
    print("SSIM rec orig:", ssim(f,u_opt, data_range = 1.0))
    print("PSNR rec orig:",psnr(u_opt,u))

    u_opt_blurred, t_opt_recursive = optTV_golden(f, blurred, tol = tol)
    print("SSIM rec recursive:", ssim(f,u_opt_blurred, data_range = 1.0))
    print("PSNR rec recursive:", psnr(u_opt_blurred,u))
    u_opt_fixed, t_opt_fixed = optTV_golden(f, avg_img, tol = tol)
    print("PSNR rec recursive:", psnr(u_opt_fixed,u))

def test_gauss_opt_sig():
    u = imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sig = 0.08
    f = add_gaussian_noise(u, sigma = sig)
    N = 100
    blur_sigs = np.linspace(0,3,N)
    u_hats = np.zeros((N,)+ f.shape)
    u_ts = np.zeros((N,)+ f.shape)
    t_list = np.linspace(0.0, 1.0, N)
    R_list = np.zeros(N)
    for i in range(1,N-1):
        print(i)
        t = t_list[i]
        u_ts[i] = ChambollePock_denoise(f, (1-t)/t, tau = 0.4, sig = 0.25, acc = True, tol = 1e-4) 
        u_hats[i] = scipy.ndimage.gaussian_filter(f,blur_sigs[i])
    Rs = np.zeros((N,N))
    for i in range(1,N-1):
        for j in range(1,N-1):
            print(i,j)
            Rs[i,j] = np.linalg.norm(u_ts[i] - u_hats[j])**2

    tt, ss = np.meshgrid(t_list, blur_sigs)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(tt, ss, Rs,
                       linewidth=0, antialiased=False)
    plt.show()

def plotFourier():
    u = imageio.imread('images/lenna.png', pilmode = 'F')
    n = 512
    fu = fft2(u)
    #plt.imsave('brick_dft.png',np.log(np.abs(fu)),cmap = 'gray')
    #plt.imsave('brick_dft_center.png', np.log(np.abs(fftshift(fu))), cmap = 'gray')
    m = 512
    window = np.outer(gaussian(n, 150), gaussian(n,150))
    plt.imshow(u*window)
    plt.show()
    fu_win = fft2(u*window)
    plt.imshow(np.abs(fftshift(fu_win)), cmap = 'gray')
    plt.show()
    u_padded = np.pad(u,(m,m), mode = 'edge')
    #plt.imsave('brick_periodic.png', u_padded, cmap = 'gray')
    #plt.imshow(u_padded, cmap = 'gray')
    #plt.show()
    fu = fft2(u_padded)
    plt.imshow(np.log(np.abs(fftshift(fu))), cmap = 'gray')
    plt.show()

def plotConvolution():

    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    n = u.shape[0]
    m = 63

    shape = (n + m - 1, n + m - 1)
    blurx_range = 15
    blury_range = 4
    k = np.zeros((m,m))
    for i in range(- blury_range//2, blury_range//2+1):
        k[i + m//2,(m//2-blurx_range):-(-blurx_range+m//2)] = 1.0
    k = k/np.sum(k)
    plt.imsave('kernel.png', k, cmap = 'gray')
    k = np.pad(k, (0,n-m), mode = 'constant')
    plt.imsave('kernel_nopad.png', k, cmap = 'gray')
    plt.imshow(k,cmap = 'gray')
    plt.show()
    fu = fft2(u)
    #plt.imshow(np.abs(fftshift(fu_win)), cmap = 'gray')
    fk = fft2(k, (n,n))
    plt.imsave('flenna_nopad.png', showFourier(fu), cmap = 'gray')
    plt.imsave('fkernel_nopad.png', showFourier(fk), cmap = 'gray')
    plt.imsave('fresult_nopad.png', showFourier(fu*fk), cmap = 'gray')

    result = np.abs(ifft2(fu*fk))
    plt.imsave('result_nopad.png', result, cmap = 'gray')
    result_padded = np.pad(result, (n,n), mode = 'wrap')
    plt.imsave('result_nopad_ext.png', result_padded, cmap = 'gray')
    
    u_pad = np.pad(u, (0,m-1), mode = 'constant')
    k_pad = np.pad(k, (0,n-1), mode = 'constant')
    plt.imsave('lenna_pad.png', u_pad, cmap = 'gray')
    plt.imsave('kernel_pad.png', k_pad, cmap = 'gray')

    fu_pad = fft2(u_pad, shape)
    fk_pad = fft2(k_pad, shape)
    plt.imsave('flenna_pad.png',  showFourier(fu_pad), cmap = 'gray')
    plt.imsave('fkernel_pad.png', showFourier(fk_pad), cmap = 'gray')
    plt.imsave('fresult_pad.png', showFourier(fu_pad*fk_pad), cmap = 'gray')

    result_pad = np.abs(ifft2(fu_pad * fk_pad))
    plt.imsave('result_pad.png', result_pad, cmap = 'gray')
    result_pad_crop = center(result_pad,n,m)
    plt.imsave('result_pad_crop.png', result_pad_crop, cmap = 'gray')
    result_pad_padded = np.pad(result_pad, shape, mode = 'wrap')
    plt.imsave('result_pad_ext.png', result_pad_padded, cmap = 'gray')

    u_pad = np.pad(u, (m//2,m//2), mode = 'edge')
    k_pad = np.pad(k, (0,n-1), mode = 'constant')
    plt.imsave('lenna_pad_edge.png', u_pad, cmap = 'gray')

    fu_pad = fft2(u_pad, shape)
    plt.imsave('flenna_pad_edge.png',  showFourier(fu_pad), cmap = 'gray')
    plt.imsave('fresult_pad_edge.png', showFourier(fu_pad*fk_pad), cmap = 'gray')

    result_pad = np.abs(ifft2(fu_pad * fk_pad))
    plt.imsave('result_pad_edge.png', result_pad, cmap = 'gray')
    result_pad_crop = result_pad[(m-1):,(m-1):]
    plt.imsave('result_pad__edge_crop.png', result_pad_crop, cmap = 'gray')
    result_pad_padded = np.pad(result_pad, shape, mode = 'wrap')
    plt.imsave('result_pad_edge_ext.png', result_pad_padded, cmap = 'gray')

def testConvolution():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    n = u.shape[0]
    m = 15

    shape = (n + m - 1, n + m - 1)
    # GAUSSIAN
    sig_blur = 1.5
    k = np.outer(signal.gaussian(m, 1*sig_blur), signal.gaussian(m, 1*sig_blur))

    k = k/np.sum(k)

    u_pad = edge_pad_and_shift(u,m)
    #
    #lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    #one = np.array([[0,0,0],[0,1,0],[0,0,0]])
    #flap = fft2(lap, shape)
    #fone = fft2(one, shape)

    #fker = fft2(k,shape)
    #fker_star = fft2(np.flip(k), shape)
    #lam = 1.0
    #lam = 0.05
    #freg = 1/(np.abs(fker)**2 - lam * flap) * fker_star
    #freg_denoise = 1/(fone - lam * flap)
    
    #plt.imsave('fgaussain.png', showFourier(fker), cmap = 'gray')
    #plt.imsave('finvgaussain.png', showFourier(1/fker), cmap = 'gray')
    #plt.imsave('QV_DFT.png', showFourier(freg), cmap = 'gray')
    #plt.imsave('QV_DFT_denoise.png', showFourier(freg_denoise), cmap = 'gray')
    #exit()
    
    sigma = 0.02
    fker = rfftn(k, shape)
    fker_star = rfftn(np.flip(k), shape)
    #flap = rfftn(lap, shape)
    #lam = 0.05
    #freg = 1/(np.abs(fker)**2 - lam * flap) * fker_star
    f = center(irfftn(rfftn(u_pad, shape)*fker),n,m)
    #plt.imsave('lenna_blur.png', f, cmap = 'gray')
    f = add_gaussian_noise(f, sigma)
    #plt.imsave('lenna_blur_noisy.png', f, cmap = 'gray')
    #print(psnr(u,f))
    print("CONDITIION NUMBER: ", np.max(np.abs(fker))/np.min(np.abs(fker)))
    ##fker = rfftn(k_padded,shape)
    #naive_deconvolve = irfftn(rfftn(np.pad(f, ((m-1)//2, (m-1)//2), 'edge'), shape)/fker)
    ##print(psnr(naive_deconvolve[:n,:n], u))
    ##naive_deconvolve = irfftn(rfftn(edge_pad_and_shift(f,m))/fker)
    #deconvolve = irfftn(rfftn(np.pad(f, ((m-1)//2, (m-1)//2), 'edge'), shape)*freg)

    ##print(psnr(deconvolve[(m-1):,(m-1):], u))
    #if(sigma == 0.0):
    #    plt.imsave('naive_deconvolve.png', naive_deconvolve[:n,:n], cmap = 'gray')
    #    plt.imsave('QV_deconvolve.png', deconvolve[(m-1):, (m-1):], cmap = 'gray')
    #else:
    #    plt.imsave('naive_deconvolve_noisy.png', naive_deconvolve[:n,:n], cmap = 'gray')
    #    plt.imsave('QV_deconvolve_noisy.png', deconvolve[(m-1):, (m-1):], cmap = 'gray')


    #fker_star = rfftn(k,shape)
    #u_opt,t_opt = optTV_golden_convolution(f,u,fker,fker_star,shape, ChambollePock_convolution_edge)
    N = 200
    #t_list = np.logspace(-1.0,0.0,N,base=100)[1:-1]
    t_list = np.flip(1 - np.logspace(-1.0,0.0,N, base = 100000000)[1:-1])
    print(t_list)
    t_opt, R_list = gridSearch_convolution(f,u,N,fker,fker_star,shape, ChambollePock_convolution_edge)
    plt.plot(t_list,R_list)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$E(t)$')
    plt.grid()
    plt.show()
    lambdas = [1e-1, 1e-2, 1e-3, 1e-4]
    lam = 1e-8
    plt.imsave('lenna_blur_noisy.png', f, cmap = 'gray')
    for lam in lambdas:
        result = ChambollePock_convolution_edge(f,lam, fker, fker_star,shape)
        plt.imsave('lenna_deblur' + str(lam) + '.png', result, cmap = 'gray')
    print(psnr(f,u))
    #print(psnr(center(result,n,m),u))
    print(psnr(result,u))
    fig,ax = plt.subplots(1, 3)
    ax[0].imshow(u, cmap="gray")
    ax[0].title.set_text("ORIGINAL")
    ax[1].imshow(f, cmap = "gray")
    ax[1].title.set_text("BLURRED")
    #ax[2].imshow(center(result,n,m), cmap = "gray")
    ax[2].imshow(result, cmap = "gray")
    ax[2].title.set_text("RESTORED")
    plt.show()
    
def testConvolutionDownsampling():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sigma = 0.1

    n = u.shape[0]
    
    m = 63
    new_m =31
    new_mdiv=new_m//2

    shape = (n + m - 1, n + m - 1)
    newshape = (n//2 + new_m - 1, n//2 + new_m - 1)
    # GAUSSIAN
    sig_blur = 1.0

    k = np.outer(signal.gaussian(m, 1*sig_blur), signal.gaussian(m, 1*sig_blur))
    #k = np.ones((m,m))
    k = k/np.sum(k)
    k_padded = np.pad(k,(0,n), 'constant')
    fker = rfftn(k_padded,shape)
    fker_star = rfftn(np.flip(k), shape)

    u_pad = edge_pad_and_shift(u,m)
    print("CONDITIION NUMBER: ", np.max(np.abs(fker))/np.min(np.abs(fker)))

    f = add_gaussian_noise(center(irfftn(rfftn(u_pad, shape)*fker),n,m), sigma)
    deconvolve = irfftn(rfftn(np.pad(f, ((m-1)//2, (m-1)//2), 'edge'), shape)/fker)
    deconvolve = deconvolve[:n,:n]
    print(psnr(deconvolve,u))
    plt.imshow(f, cmap = 'gray')
    plt.title("BLURRED")
    plt.show()
    
    plt.imshow(deconvolve[:n,:n],cmap='gray')
    plt.title("NAIVE DECONVOLVE")
    plt.show()

    xk = np.arange(0,m,1)
    yk = np.arange(0,m,1)
    new_xk = np.arange(1,m,2)
    new_yk = np.arange(1,m,2)

    xf = np.arange(0,n,1)
    yf = np.arange(0,n,1)
    new_xf = np.arange(0,n,2)
    new_yf = np.arange(0,n,2)

    interpk = scipy.interpolate.interp2d(xk,yk,k, kind='linear')
    k_down= interpk(new_xk,new_yk)
    # NEED TO SCALE DOWNSAMPLED KERNEL
    k_down = k_down/np.sum(k_down)
    interpf = scipy.interpolate.interp2d(xf,yf,f, kind='linear')
    f_down = interpf(new_xf,new_yf)
    ff_down = rfftn(np.pad(f_down, (new_mdiv, new_mdiv)), newshape)

    plt.imshow(f_down, cmap = 'gray')
    plt.title("DOWNSAMPLED")
    plt.show()

    fk_down = rfftn(k_down, newshape)
    print("CONDITIION NUMBER: ", np.max(np.abs(fk_down))/np.min(np.abs(fk_down)))
    deconvolve_down =  irfftn(ff_down/fk_down, newshape)
    u_down = deconvolve_down[:n//2,:n//2]
    plt.imshow(u_down, cmap = 'gray')
    plt.title("DOWNSAMPLED DECONVOLVE")
    plt.show()
    interpf = scipy.interpolate.interp2d(new_xf,new_yf,u_down, kind='linear')
    u_up = interpf(xf,yf)
    print(psnr(u_up,u))
    plt.imshow(u_up, cmap = 'gray')
    plt.title('UPSAMPLED')
    plt.show()

    u_opt,t_opt = optTV_golden_convolution(f,u_up,fker,fker_star,shape,t_left = 0.98, tol = 1e-3)
    print(psnr(u_opt,u))
    plt.imshow(u_opt, cmap= 'gray')
    plt.title("OPTTV")
    plt.show()

    return 0 

def testConvolutionOperator():
    u_1 =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    n = u_1.shape[0]
    m = 15

    shape = (n + m - 1, n + m - 1)
    # GAUSSIAN
    sig_blur = 1.0
    k = np.outer(signal.gaussian(m, 100*sig_blur), signal.gaussian(m, 1*sig_blur))
    fk = rfftn(k,shape)
    fk_star = rfftn(np.flip(k),shape)
    sigma = 0.3
    u_2 = add_gaussian_noise(np.zeros((n,n)),sigma)
    Au_1 = center(irfftn(rfftn(u_1,shape)*fk), n, m)
    A_staru_2 = center(irfftn(rfftn(u_2,shape)*fk_star), n, m)
    print(np.sum(np.multiply(Au_1, u_2)))
    print(np.sum(np.multiply(u_1, A_staru_2)))

def plotRinging():
    def ringingFunc(x,y):
        if (x**2 + y**2 < 0.77):
            return np.exp(x**2 + y**2)
        else:
            return np.exp(x**2 + y**2)/2

    n = 64
    h = 1/n
    f = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            f[i,j] = ringingFunc(i*h,j*h)
    plt.imsave('test.png', f)
    interp_list = ['nearest', 'sinc', 'bilinear', 'bicubic']
    for interp in interp_list:
        plt.imshow(f, interpolation = interp, cmap = 'gray')
        plt.show()

def forPresentation():
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    """
    GRID SEARCHES PLOT
    """
    N = 100
    print("NOISY PSNR:",psnr(u,f))
    #blur_sigs = np.linspace(0,3,100)
    #blurs = np.zeros(100)
    #i = 0
    #for blur_sig in blur_sigs:
    #    blurred = scipy.ndimage.gaussian_filter(f,blur_sig)
    #    blurs[i] = psnr(blurred,u)
    #    i+=1
    #best_blur_sig = blur_sigs[np.argmax(blurs)]
    #blurred = scipy.ndimage.gaussian_filter(f,best_blur_sig)
    #blurred = scipy.ndimage.gaussian_filter(f,1.0)


    t_list = np.linspace(0.0, 1.0, N)
    t_opt, R_list = gridSearch(f,u,N, f2py_cp_denoise, plot = False, tol = 1.0e-8)
    plt.plot(t_list[1:-1], R_list)
    plt.grid()
    plt.legend()
    plt.title(r'Loss function for an example TV problem')
    plt.xlabel(r'$t$')
    plt.ylabel(r'R(t)')
    plt.savefig('TVR.png')
    plt.show()
    t_list = np.linspace(0.0, 1.0, N)
    t_opt, R_list = gridSearch(f,u,N, f2py_quadratic_denoise, plot = False, tol = 1.0e-8)
    plt.plot(t_list[1:-1], R_list)
    plt.grid()
    plt.legend()
    plt.title(r'Loss function for an example QV problem')
    plt.xlabel(r'$t$')
    plt.ylabel(r'R(t)')
    plt.savefig('QVR.png')
    plt.show()

    tol = 1e-4
    u_opt, t_opt  = optTV_golden(f, u, tol = tol)
    print("OPTIMAL PSNR:", psnr(u,u_opt))
    u_opt_blurred, t_opt_blurred  = optTV_golden(f, blurred, tol = tol)

    fig,ax = plt.subplots(1, 3)
    ax[0].imshow(f, cmap="gray")
    ax[0].title.set_text("Noisy, PSNR = " + '{:.2f}'.format(psnr(u,f)))
    ax[1].imshow(blurred, cmap = "gray")

    ax[1].title.set_text("Blurred, PSNR = " + '{:.2f}'.format(psnr(u,blurred)))# + psnr(u,blurred))
    #ax[2].imshow(center(result,n,m), cmap = "gray")
    ax[2].imshow(u_opt_blurred, cmap = "gray")

    ax[2].title.set_text("Blurred, PSNR = " + '{:.2f}'.format(psnr(u,u_opt_blurred)))# + psnr(u,u_opt_blurred))
    plt.show()
    plt.imsave('noisy.png', f, cmap = 'gray')
    plt.imsave('blurred.png', blurred, cmap = 'gray')
    plt.imsave('u_opt_blurred.png', u_opt_blurred, cmap = 'gray')

def testTV():
    n = 64
    u = np.zeros((n,n))
    # CONSTANT IMAGE
    u[0:n//2,:n//2] = 0.7
    u[0:n//2,n//2:] = 0.3
    
    #Checkerboard image
    for i in range(n//2, n):
        for j in range(0,n,2):
            u[i,i%2 + j] = 0.5
    sigma = 0.3
    f = add_gaussian_noise(u, sigma)
    N = 100
    plt.imshow(f)
    plt.show()
    gridSearch(f,u,N)

def testDownsampledConvolution():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    #u_opt, t_opt = optTV_golden(f,u, tol= 1e-4)
    #print(psnr(u,u_opt))

    m = u.shape[0]
    n = 3
    N = 10
    M = 3
    pert = 0.01
    image_array = createResampledArray(f, n, N)
    new_image_array = np.zeros((2,m,m))
    avg_img = np.average(image_array, axis = 0) 
    print(psnr(avg_img,u))
    u_opt, t_opt = optTV_golden(f,avg_img, tol = 1e-4) 
    print(psnr(u_opt, u))
    new_image_array[0] = u_opt
    t_pert = t_opt + pert
    new_image_array[1] = ChambollePock_denoise(f, (1-t_pert)/t_pert, tau = 0.4, sig = 0.25, acc = True, tol = 1e-4)
    new_image = 0.9*new_image_array[0] + 0.1*new_image_array[1]
    #new_image = np.average(new_image_array, axis = 0)

    print(psnr(new_image, u))
    u_opt, t_opt = optTV_golden(f,new_image, tol = 1e-4) 
    print(psnr(u_opt,u))
    exit()
    
    
    denoise_array = np.zeros((M,m,m))
    #gridSearch(f, u, 100)
    #t_list = np.linspace(0.85,0.98,N)
    #for i,t in enumerate(t_list):
    #    denoise_array[i] = ChambollePock_denoise(f, (1-t)/t, tau = 0.4, sig = 0.25, acc = True, tol = 1e-4)
    avg_img = np.average(denoise_array, axis = 0)
    print(psnr(u,f))
    print(psnr(avg_img,u))
    plt.imshow(avg_img, cmap = 'gray')
    plt.show()
    u_opt, t_opt = optTV_golden(f,avg_img, tol= 1e-4)
    plt.imshow(u_opt, cmap = 'gray')
    plt.show()

def testGCV():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    n = u.shape[0]
    temp = 1/(n*n)
    sigma = 0.1
    f = add_gaussian_noise(u, sigma)
    N =50
    t_list = np.linspace(0.01,1.0,N)
    GCV_list = np.zeros(N)

    for i,t in enumerate(t_list):
        print(i, t)
        q = add_gaussian_noise(np.zeros((n,n)),1.0)
        trace = np.sum(np.multiply(q, ChambollePock_denoise(q,t, tau = 0.25, sig = 0.25, acc = True, tol = 1e-4)))
        u_t = ChambollePock_denoise(f,t, tau = 0.25, sig = 0.25, acc = True, tol = 1e-4)
        print(np.linalg.norm(u_t - f,2))
        print(temp*trace)
        GCV_list[i] = (temp * np.linalg.norm(u_t - f,'fro')**2)/((1 - temp * trace)**2)
        print(GCV_list[i])
    plt.plot(t_list, GCV_list)
    plt.show()

def testquadratcic():
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/(255.0)
    n = u.shape[0]
    
    sigmas = [0.05, 0.1]
    marker = ['r-', 'g-','b-']
    marker_hat = ['r--', 'g--','b--']
    N = 200
    
    t_list = np.linspace(0.001,0.999,N)
    R_list = np.zeros((len(sigmas),N))
    R_hat_list = np.zeros((len(sigmas),N))
    #j = 0
    #for sig in sigmas:
    #    f = add_gaussian_noise(u, sig)
    #    u_hat, t_QV = optTV(f, u, f2py_quadratic_denoise)
    #    for i, t in enumerate(t_list):
    #        result = f2py_cp_denoise(f, (1-t)/t)
    #        R_list[j,i] = np.linalg.norm(result- u)**2
    #        R_hat_list[j,i] = np.linalg.norm(result- u_hat)**2
    #    plt.plot(t_list,R_hat_list[j], marker_hat[j])
    #    plt.plot(t_list,R_list[j], marker[j])
    #    j+=1
    #plt.show()
    f = add_gaussian_noise(u, 0.1)
    u_QV, t_QV = optTV(f, u, f2py_quadratic_denoise)
    u_TV, t_TV = optTV(f, u_QV, f2py_cp_denoise)
    u_opt, t_opt = optTV(f,u, f2py_cp_denoise)
    plt.imsave('QV_denoise_opt.png', u_QV, cmap = 'gray')
    plt.imsave('TV_denoise_opt.png', u_TV, cmap = 'gray')
    plt.imshow(u_QV)
    plt.show()
    plt.imshow(u_TV)
    plt.show()
    plt.imshow(u_opt)
    plt.show()

def counterexample():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/(3*255.0)
    #u[:,:] = 0.0
    n = u.shape[0]

    #u[:n//2,:] = 0.5
    #for i in range(n//2, n):
    #    for j in range(0,n,2):
    #        u[i,i%2 + j] = 0.5

    sigma = 0.1
    f = add_gaussian_noise(u, sigma)
    u_hat = 3*counterexample(f,0.8167)
    N = 100
    #plt.imshow(f,cmap = 'gray')
    #plt.show()

    #result = quadraticRegularizer_convolution(f,0.00001, fker, fker_star, tol = 1.0e-5)
    #plt.imshow(result, cmap = 'gray')
    #plt.show()
    #exit()
    t_list = np.linspace(0.001,0.99,N)
    psnr_list = np.zeros(N)
    for i, t in enumerate(t_list):
        result = quadraticRegularizer_denoise(f,(1-t)/t, tol = 1.0e-5)
        psnr_list[i] = np.linalg.norm(result- u_hat)**2
        print(i)
    t_opt = t_list[np.argmin(psnr_list)]
    print(t_opt)
    print((1-t_opt)/t_opt)
    u_hat = quadraticRegularizer_denoise(f, (1 - t_opt)/t_opt, tol = 1.0e-10)
    print(psnr(u_hat,u))
    plt.plot(t_list,psnr_list)
    plt.show()
    plt.imshow(u_hat, cmap = 'gray')
    plt.show()

    u_opt, t_opt = optTV_golden(f,u_hat, tol = 1.0e-4)
    u_true, t_true = optTV_golden(f,u, tol = 1.0e-4)

    print(psnr(u_true,u))
    print(psnr(u_opt,u))
    plt.imshow(u_opt, cmap='gray')
    plt.show()
    plt.imshow(u_true, cmap='gray')
    plt.show()

def testquadraticdeconvolve():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/(255.0)
    n = u.shape[0]

    sig_blur = 1.0
    plt.show()
    m = 31
    shape = (n + m - 1, n + m - 1)
    k = np.outer(signal.gaussian(m, 1*sig_blur), signal.gaussian(m, 1*sig_blur))
    #k = np.ones((m,m))
    k = k/np.sum(k)
    fker = rfftn(k, shape) 
    fker_star = rfftn(np.flip(k), shape)
    sigma = 0.01
    f = add_gaussian_noise(center(irfftn(rfftn(edge_pad_and_shift(u,m), shape)*fker),n,m), sigma)
    plt.imshow(f)
    plt.show()
    N = 100
    t_list = np.linspace(0.5,0.99,N)
    psnr_list = np.zeros(N)
    for i, t in enumerate(t_list):
        print(i)
        result = quadraticRegularizer_convolution(f, (1-t)/t, fker, fker_star, tol = 1.0e-8)
        psnr_list[i] = np.linalg.norm(result- u)**2
    t_opt = t_list[np.argmin(psnr_list)]
    print(t_opt)
    u_hat = quadraticRegularizer_convolution(f, (1 - t_opt)/t_opt, fker, fker_star, tol = 1.0e-10)
    u_opt, t_opt = optTV_golden_convolution(f,u_hat, fker, fker_star, shape,tol = 1.0e-4)
    u_true, t_true = optTV_golden_convolution(f,u, fker, fker_star, shape,tol = 1.0e-4)
    print(psnr(u_true,u))
    print(psnr(u_opt,u))
    plt.imshow(u_opt, cmap='gray')
    plt.show()
    plt.imshow(u_true, cmap='gray')
    plt.show()



    #print(psnr(u_hat,u))
    #plt.plot(t_list,psnr_list)
    #plt.show()
    #plt.imshow(u_hat, cmap = 'gray')
    #plt.show()
    #t_opt = 0.8
    #u_hat = quadraticRegularizer_convolution(f, (1 - t_opt)/t_opt, fker, fker_star, tol = 1.0e-10)
    #plt.imshow(u_hat)
    #plt.show()

def testf2py():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    u = add_gaussian_noise(u,sigma)

    n = u.shape[0]
    du = np.zeros((2,n,n), dtype=np.float32, order = 'F')
    lam = 10
    tol1 = 1.0e-8
    tol2 = 1.0e-8
    t = time.time()
    res1 = f2py_quadratic_denoise(u, lam, tol = tol1)
    res2 = f2py_cp_denoise(u, lam, tol = tol2)

    plt.imshow(res1)

    plt.show()
    #t = time.time()
    #result = ChambollePock_denoise(u, lam, tau = 0.25, sig = 0.25, theta = 1.0, acc = False)
    #print(time.time() - t)
    #plt.imshow(result - res)
    #plt.show()
    #plt.imshow(result)
    #plt.show()
    #plt.imshow(res, cmap = 'gray')
    #plt.show()

def testparameterselection():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.3
    f = add_gaussian_noise(u,sigma)
    N = 100
    gridSearch(f, u, N, f2py_quadratic_denoise, tol = 1.0e-16)
    #u_opt, t_opt = discrepancy_rule(f,sigma,f2py_cp_denoise, tol = 1.0e-8)
    #print(t_opt)
    #print(psnr(u,u_opt))
    #u_opt, t_opt = L_curve(f,f2py_cp_denoise, tol = 1.0e-8)
    #print(t_opt)
    #print(psnr(u,u_opt))
    #u_opt, t_opt = quasi_optimality(f,f2py_cp_denoise, tol = 1.0e-8)
    #print(t_opt)
    #print(psnr(u,u_opt))
    #u_opt, t_opt = optTV(f,u, f2py_cp_denoise, tol = 1.0e-8)
    #print(t_opt)
    #print(psnr(u,u_opt))
    #u_opt,t_opt = GCV(f, f2py_cp_denoise,tol = 1.0e-8)
    print(t_opt)
    print(psnr(u,u_opt))
    plt.imshow(u_opt)
    plt.show()

def downsamplingexperiment():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.2
    f = add_gaussian_noise(u,sigma)
    n = f.shape[0]
    coords = np.zeros((2,n)) 
    coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1)

def testnewestimator():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)

    new_n = 256
    alpha = new_n/f.shape[0]
    ds_number = 1

    discrepancy_rule_set_noise = lambda f, method, **method_kwargs: discrepancy_rule(f, method,sigma*0.5, lam_init = 0.5, **method_kwargs)
    quasi_optimality_ = lambda f, method, **method_kwargs: quasi_optimality(f, method, lam_init = 0.5, **method_kwargs)

    #u_hat = createEstimator(f,new_n,ds_number, f2py_cp_denoise, discrepancy_rule_set_noise, tol=1.0e-7)
    u_hat = createEstimator(f,new_n,ds_number, f2py_cp_denoise, L_curve, tol=1.0e-8)
    plt.imshow(u_hat)
    plt.show()

    print(psnr(u,f))
    print(psnr(u,u_hat))
    u_opt, t_opt = optTV(f, u, f2py_cp_denoise, tol = 1.0e-8)
    u_dp, t_dp = L_curve(f,f2py_cp_denoise,tol=1.0e-8)
    print(psnr(u_dp,u))
    u_rec, t_rec = optTV(f, u_hat, f2py_cp_denoise, tol = 1.0e-8)
    print(psnr(u_opt,u))
    print(psnr(u_rec,u))
    plt.imshow(u_rec, cmap='gray')
    plt.show()

def plotnoise():
    u_1 = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    u_2 = np.array(imageio.imread('images/lenna256.jpg', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    f_1 = add_gaussian_noise(u_1,sigma)
    f_2 = add_gaussian_noise(u_2,sigma)
    print(psnr(f_1,u_1))
    print(psnr(f_2,u_2))

    plt.imsave('noisy01512.png', f_1, cmap = 'gray', format = 'png')
    plt.imsave('noisy01256.png', f_2, cmap = 'gray', format = 'png')

    f_1 = np.copy(u_1)
    f_2 = np.copy(u_2)
    # SALT AND PEPPER NOISE
    n = u_1.shape[0]
    r_1 = numpy.random.uniform(0.0,1.0,np.prod(u_1.shape)).reshape(u_1.shape)
    for i in range(u_1.shape[0]):
        for j in range(u_1.shape[1]):
            if(r_1[i,j] < sigma/2):
                f_1[i,j] = 0.0
            elif(r_1[i,j] > 1 - sigma/2):
                f_1[i,j] = 1.0
    plt.imsave('sp01512.png', f_1, cmap= 'gray', format = 'png')

def testquadraticparameterselection():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    print(psnr(u,f))
    #u, t = L_curve(f, f2py_cp_denoise,lam_init = 1.0, q = 0.9, plot = True)
    #discrepancy_rule_set_noise = lambda f, method, **method_kwargs: DP_secant(f, method,sigma, **method_kwargs)
    #u_opt, t_opt = optTV(f,u, f2py_cp_denoise)
    #u_alt, t_alt = optTV(f,u,f2py_quadratic_denoise)
    #print(t_opt)
    #print(t_alt)
    #print(psnr(u_opt,u))
    #print(psnr(u_alt,u))
    methods = [LC_golden]
    for method in methods:
        print(str(method))
        u_rec, t_rec = method(f, f2py_quadratic_denoise, t_left = 0.1, t_right = 1.0)
        u_tv, t_tv = optTV(f,u_rec, f2py_cp_denoise, tol = 1.0e-8)
        print(t_rec)
        print(psnr(u_rec,u))
        print(psnr(u_tv,u))
    #u_opt, t_opt = optTV(f,u, f2py_quadratic_denoise, tol = 1e-8)
    #print(t_opt)
    #print(psnr(u_opt,u))
    ##gridsearch(f, u_opt, f2py_cp_denoise
    #u_TV, t_TV = optTV(f,u_opt, f2py_cp_denoise, tol = 1e-7)
    #u_optTV, t_optTV = optTV(f,u, f2py_cp_denoise, tol = 1e-7)

    #plt.imshow(u_opt)
    #plt.show()
    #plt.imshow(u_rec)
    #plt.show()

def plotfouriernoise():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    n = 512
    sigma = 0.1
    N = 1
    noisy = np.zeros((N,n,n))
    #noisy_ds = np.zeros((N,new_n,new_n)) 
    for i in range(N):
        noisy[i] = add_gaussian_noise(np.ones((n,n))*0.5,sigma)
    f = u + noisy[i] - 0.5
    ff = fftn(f)
    fu = fftn(u)
    vmin = min(np.min(u), np.min(noisy[0]-0.5),np.min(f)) 
    vmax = max(np.max(u), np.max(noisy[0]-0.5),np.max(f)) 
    plt.imsave('lenna_alt.png', u, cmap = 'gray', vmin = vmin, vmax=vmax)
    plt.imsave('noise.png', noisy[0] - 0.5, cmap = 'gray', vmin = vmin, vmax=vmax)
    plt.imsave('lenna_alt_noisy.png', f, cmap = 'gray', vmin = vmin, vmax=vmax)
    vmin = min(np.log((np.min(np.abs(f)), np.min(np.abs(ff)), np.min(np.abs(fu)))))
    vmax = max(np.log((np.max(np.abs(f)), np.max(np.abs(ff)), np.max(np.abs(fu)))))
    print(vmin, vmax)
    fnoisy = fftn(noisy[0] - 0.5)
    plt.imsave('flenna_alt.png',showFourier(fu), cmap = 'gray', vmin = vmin, vmax = vmax)
    plt.imsave('flenna_noisy_alt.png',showFourier(ff), cmap = 'gray', vmin = vmin, vmax = vmax)
    plt.imsave('fnoise.png', showFourier(fnoisy), cmap = 'gray', vmin = vmin, vmax = vmax)

    new_n = 511
    #coords = np.zeros((2,n)) 
    #coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1) # The original image exitst on [0,n-1]X[0,n-1]

    #coords_ds = np.zeros((2,n))
    #coords_ds[0,:new_n] = np.linspace(0.0, n - 1.0, new_n, endpoint = True) 
    #coords_ds[1,:new_n] = np.linspace(0.0, n - 1.0, new_n, endpoint = True)

    #for i in range(N):
    #    interpf = scipy.interpolate.interp2d(coords[0], coords[1], noisy[i], kind ='linear')
    #    noisy_ds[i] = interpf(coords_ds[0,:new_n], coords_ds[0,:new_n])
    #    #plt.imshow(noisy_ds[i])
    #    #plt.show()
    #std = np.std(noisy_ds, axis = 0)**2
    #plt.imshow(std)
    #plt.show()
    #plt.imshow(showFourier(fftn(std), log = True))
    #plt.show()

def downsamplingexperiment():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    n = f.shape[0]
    coords = np.zeros((2,n)) 
    coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1) # The original image exitst on [0,n-1]X[0,n-1]
    interpu = scipy.interpolate.interp2d(coords[0], coords[1], u, kind ='linear')
    interpf = scipy.interpolate.interp2d(coords[0], coords[1], f, kind ='linear')
    coords_ds = np.zeros((2,n))
    stds = np.zeros(n)
    for new_n in range(n//16, 512):
        print(new_n)
        coords_ds[0,:new_n] = np.linspace(0.5, n - 1.5, new_n, endpoint = True) 
        coords_ds[1,:new_n] = np.linspace(0.5, n - 1.5, new_n, endpoint = True)
        u_ds =interpu(coords_ds[0,:new_n], coords_ds[0,:new_n])
        f_ds =interpf(coords_ds[0,:new_n], coords_ds[0,:new_n])
        stds[new_n] = np.std(f_ds - u_ds) 
    plt.plot(np.arange(n//16,n),stds[n//16:])
    plt.show()
    return 0

def plotparametermethods():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    u, t = optTV(f,u, f2py_cp_denoise, tol = 1.0e-8)
    print(t)
    #u,t = discrepancy_rule(f, f2py_cp_denoise, sigma, lam_init = 1.0, plot = True, tol = 1.0e-8)
    #u,t = discrepancy_rule(f, f2py_quadratic_denoise, sigma, lam_init = 50.0, plot = True, tol = 1.0e-8)
    #u,t = L_curve(f, f2py_quadratic_denoise, lam_init = 50.0, plot = True, tol = 1.0e-8)
    #u,t = L_curve(f, f2py_cp_denoise, lam_init = 10.0, plot = True, tol = 1.0e-8)
    #u,t = quasi_optimality(f, f2py_cp_denoise, lam_init = 10.0, q = 0.95,plot = True, tol = 1.0e-8)
    u,t = quasi_optimality(f, f2py_quadratic_denoise, lam_init = 10.0, q = 0.95,plot = True, tol = 1.0e-8)
    #u, t = DP_secant(f, f2py_cp_denoise, sigma, tol = 1.0e-8)
    #u, t = DP_bisection(f, f2py_cp_denoise, sigma, tol = 1.0e-8)
    #u, t = DP_rf(f, f2py_cp_denoise, sigma, tol = 1.0e-8)
    print(t)

def import_cifar():
    import pickle
    with open('images/cifar/data_batch_1', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    dict[b'data'] = dict[b'data'].reshape((10000,32,32,3), order = 'F')
    dict[b'data'] = np.dot(dict[b'data'][...,:3], [0.2989, 0.5870, 0.1140])/255.0


    return dict
def ploteigenvalues():
    dict = import_cifar()
    labels = dict[b'labels']
    label_indice = [i for i,label in enumerate(labels) if label == 4]
    data = dict[b'data'][label_indice]
    N = 999
    data_noisy = np.zeros_like(data)
    sigma = 0.05
    for i in range(N):

       data_noisy[i] = add_gaussian_noise(data[i], sigma) 

    for i in range(10):
        plt.imsave('deer_noisy' + str(i) + ".png",data_noisy[i].T, cmap = 'gray')

    S = np.zeros((data.shape[1]*data.shape[2], data.shape[1]*data.shape[2]))
    S_noisy = np.zeros((data.shape[1]*data.shape[2], data.shape[1]*data.shape[2]))
    for i in range(N):
        S += 1.0/N * np.outer(data[i,:,:], data[i,:,:])
        S_noisy+= 1.0/N * np.outer(data_noisy[i,:,:], data_noisy[i,:,:])

    eigv, eigvectors = linalg.eigh(S)
    eigv_noisy, eigvectors_noisy = linalg.eigh(S_noisy)
    eigv = eigv[::-1]
    eigv_noisy = eigv_noisy[::-1]
    plt.semilogy(eigv, label = "True (Noiseless)")
    plt.semilogy(eigv_noisy, label ="Empirical (Noisy)")
    plt.title("Eigenvalues of true and empirical covariance matrices")
    plt.legend()
    plt.show()
    for i in range(10):
        plt.imsave('eigdeer' + str(i) + ".png",eigvectors[i].reshape((32,32)).T, cmap = 'gray')
        plt.imsave('eigdeer_noisy' + str(i) + ".png",eigvectors_noisy[i].reshape((32,32)).T, cmap = 'gray')
    deer = data[0].reshape(32*32)
    norm_deer = np.linalg.norm(deer)
    deer_noisy = data_noisy[0].reshape(32*32)
    print(np.linalg.norm(deer_noisy - deer)/np.linalg.norm(deer))
    psnr_list = np.zeros(32*32)
    psnr_list_noisy = np.zeros(32*32)
    psnr_list_noisy_true = np.zeros(32*32)
    projdeer = np.zeros(32*32)
    projdeer_noisy = np.zeros(32*32)
    projdeer_noisy_true = np.zeros(32*32)
    save_list = [1, 16, 32, 64, 128, 256, 512, 768, 1023]
    for i in range(1,32*32):
        eigvs = eigvectors[:,-i:].T[::-1].T
        eigvs_noisy = eigvectors_noisy[:,-i:].T[::-1].T
        P = eigvs.dot(eigvs.T)
        P_noisy = eigvs_noisy.dot(eigvs_noisy.T)
        projdeer = np.dot(deer,P).T
        projdeer_noisy = np.dot(deer_noisy,P_noisy).T
        projdeer_noisy_true = np.dot(deer_noisy,P).T
        if (i in save_list):
            plt.imsave('projdeer' + str(i) + ".png", projdeer.reshape((32,32)).T, cmap = 'gray') 
            plt.imsave('projdeer_noisy' + str(i) + ".png", projdeer_noisy.reshape((32,32)).T, cmap = 'gray') 
            plt.imsave('projdeer_noisy_true' + str(i) + ".png", projdeer_noisy_true.reshape((32,32)).T, cmap = 'gray') 
        psnr_list[i] = np.linalg.norm(projdeer-deer)/norm_deer
        psnr_list_noisy[i] = np.linalg.norm(projdeer_noisy-deer)/norm_deer
        psnr_list_noisy_true[i] = np.linalg.norm(projdeer_noisy_true-deer)/norm_deer
    plt.plot(psnr_list[1:], label = "Noiseless, true projection")
    plt.plot(psnr_list_noisy[1:], label = "Noisy, empirical projection")
    plt.plot(psnr_list_noisy_true[1:], label = "Noisy, true projection")
    plt.title(r"$\|\hat{u} - u\|/\|u\|$")
    plt.legend()
    plt.show()

def ploteigenvalues_():
    dict = import_cifar()
    labels = dict[b'labels']
    label_indice = [i for i,label in enumerate(labels) if label == 4]
    data = dict[b'data'][label_indice]
    N = 100
    sigma = 0.01
    deer = data[0]
    deer_noisy = np.zeros((N, 32,32))
    S = np.zeros((data.shape[1]*data.shape[2], data.shape[1]*data.shape[2]))
    for i in range(N):
        deer_noisy[i] = add_gaussian_noise(deer, sigma)
        S += 1.0/N * np.outer(deer_noisy[i,:,:], deer_noisy[i,:,:])
    eigv, eigvectors = linalg.eigh(S)
    plt.semilogy(eigv)
    plt.show()

    eigvs = eigvectors[:,-1:].T[::-1].T
    P = eigvs.dot(eigvs.T)

    basis_img = modified_gram_schmidt(deer_noisy)
    print(basis_img.shape)

    test_deer = deer_noisy[0].reshape(32*32)
    true_proj = np.sum(np.multiply(test_deer.reshape((32,32)),deer))/(np.sum(np.multiply(deer,deer))) * deer
    proj_deer = np.dot(test_deer,P).reshape((32,32)).T
    gram_deer = np.sum([np.sum(np.multiply(basis_img[i],test_deer.reshape((32,32))))*basis_img[i] for i in range(N)], axis = 0)
    avg_img = np.average(deer_noisy, axis = 0)
    avg_proj = np.sum(np.multiply(test_deer.reshape((32,32)),avg_img))/(np.sum(np.multiply(avg_img,avg_img))) * avg_img
    plt.imshow(avg_img)
    plt.show()
    plt.imshow(-eigvs.reshape((32,32)))
    plt.show()
    print(avg_img/np.linalg.norm(avg_img, 'fro'))
    print(  -eigvs.reshape((32,32))/np.linalg.norm(eigvs.reshape((32,32)), 'fro'))
    print(deer)
    print(np.linalg.norm(np.abs(eigvs.reshape((32,32)))/np.linalg.norm(eigvs.reshape((32,32)), 'fro') - avg_img/np.linalg.norm(avg_img, 'fro')))
    #print(psnr(deer, avg_img))
    #print(psnr(deer, true_proj))
    #print(psnr(deer,gram_deer))
    #print(psnr(deer,proj_deer.T))
    #print(psnr(deer,avg_proj))
    #print(np.linalg.norm(proj_deer.T - avg_proj))
    #print(psnr(deer,test_deer.reshape((32,32))))

def plotprojections():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigmas = [0.05, 0.075, 0.1]
    N = 100
    fs = np.zeros((len(sigmas), N, u.shape[0], u.shape[1]))
    avg_img = np.zeros_like(u)
    psnr_list = np.zeros((len(sigmas), N))
    psnr_list_proj = np.zeros(N)
    for i in range(N):
        for j,sig in enumerate(sigmas):
            fs[j,i,:,:] = add_gaussian_noise(u,sig)
            avg_img = np.average(fs[j,:(i+1)], axis = 0)
            psnr_list[j,i] = psnr(avg_img, u)
        #plt.imshow(avg_img)
        #plt.show()
        #psnr_list_proj[i] = psnr(np.sum(np.multiply(fs[i],avg_img))/np.sum(np.multiply(avg_img,avg_img))*avg_img, u)
    
    print(np.sum(np.multiply(fs[0,0,:,:],u))/np.sum(np.multiply(u,u)) * u)
    print(np.sum(np.multiply(fs[1,0,:,:],u))/np.sum(np.multiply(u,u)) * u)
    print(np.sum(np.multiply(fs[2,0,:,:],u))/np.sum(np.multiply(u,u)) * u)
    #print(np.sum(np.multiply(fs[0],u)))
    #print(np.sum(np.multiply(u,u)))
    #plt.hlines(psnr(proj,u), xmin = 0, xmax = N)
    for i in range(len(sigmas)):
        plt.plot(psnr_list[i], label = r'$\sigma = $' + str(sigmas[i]))
    #plt.plot(psnr_list)
    #plt.plot(psnr_list_proj)
    plt.ylabel('PSNR')
    plt.xlabel('N')
    plt.title('PSNR value between original image and mean estimator')
    plt.legend()
    plt.show()

def plotprojectionparameter():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigmas = [0.05]
    N = 1
    fs = np.zeros((len(sigmas), N, u.shape[0], u.shape[1]))
    M = 200
    t_list = np.linspace(0.0,1.0,M)
    for i,sig in enumerate(sigmas):
        for j in range(N):
            fs[i,j,:,:] = add_gaussian_noise(u,sig)
        #L_curve(fs[i,0,:,:], f2py_cp_denoise, lam_init = 10.00, q = 0.9, plot = True,tol = 1.0e-10)
        #GCV_trace(fs[i,0,:,:], f2py_quadratic_denoise,lam_init = 10.0, tol = 1.0e-10)
        #quasi_optimality(fs[i,0,:,:], f2py_cp_denoise,lam_init = 100.0, plot = True,tol = 1.0e-10)
        u_QV, t_QV = GCV_golden(fs[i,0,:,:], f2py_quadratic_denoise)
        #u_QV, t_QV = optTV(fs[i,0,:,:], u,f2py_quadratic_denoise)
        #u_QV, t_QV = DP_secant(fs[i,0,:,:], f2py_quadratic_denoise,noise_std = sig)
        #u_QV, t_QV = QOC(fs[i,0,:,:], f2py_quadratic_denoise, lam_init = 10.0, tol = 1e-10)
        plt.imsave('lenna_denoise_QV_GCV.png', u_QV, cmap = 'gray')
        print(psnr(u_QV,u))

        u_TV, t_TV = optTV(fs[i,0,:,:],u_QV,f2py_cp_denoise)
        plt.imsave('lenna_denoise_QVTV_GCV.png', u_TV, cmap = 'gray')
        print(t_TV)
        print(psnr(u_TV,u))
        

def plotupsampling():
    u = np.array(imageio.imread('images/pikachu.png', pilmode = 'F')/(255.0), order = 'F')
    n = u.shape[0]
    m = 3*n
    coords = np.zeros((2,n)) 
    coords[0] = np.arange(0,n,1); coords[1] = np.arange(0,n,1)
    nn = nearest(u, ((m,m)))
    plt.imsave('pikachu_nn.png', nn, cmap = 'gray')
    downsamplelist = ['linear', 'cubic']
    for ds in downsamplelist:
        interpu = scipy.interpolate.interp2d(coords[0], coords[1], u, kind =ds)
        newx = np.linspace(0,n,m)
        f = interpu(newx,newx)
        plt.imsave('pikachu' + ds + '.png', f, cmap = 'gray')

def thedenoisingexperiment():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    n = u.shape[0]
    sigmas = [0.05, 0.1]
    M = 1
    noisy_array = np.zeros((len(sigmas), n,n))
    # This code is so horrible
    u_DPB = np.zeros((len(sigmas), n,n), order = 'F')
    t_DPB = np.zeros(len(sigmas))
    u_DPS = np.zeros((len(sigmas), n,n), order = 'F')
    t_DPS = np.zeros(len(sigmas))
    u_LC = np.zeros((len(sigmas), n,n), order = 'F')
    t_LC = np.zeros(len(sigmas))
    u_QOC = np.zeros((len(sigmas), n,n), order = 'F')
    t_QOC = np.zeros(len(sigmas))
    u_GCV = np.zeros((len(sigmas), n,n), order = 'F')
    t_GCV = np.zeros(len(sigmas))

    #for i, sig in enumerate(sigmas):
    #    print("sigma = ", sig)
    #    noisy_array[i,:,:] = add_gaussian_noise(u, sig)
    #    u_opt, t_opt = optTV(noisy_array[i,:,:], u , f2py_cp_denoise, t_left = 0.8)
    #    print("===================== OPT ===================")
    #    print("PNSR:", psnr(u_opt, u)) 
    #    print("t:", t_opt)

    #    print("===================== DPS ===================")
    #    time_DP = time.time()
    #    for j in range(M):
    #        u_DPS[i], t_DPS[i] = DP_secant(noisy_array[i,:,:],f2py_cp_denoise, noise_std = sig, t_0 = 1.0, t_1 = 0.99)
    #    print("TIME:", (time.time() - time_DP)/M)
    #    print("PSNR:", psnr(u_DPS[i],u))
    #    print("|t - t_opt|:", np.abs(t_DPS[i] - t_opt))
    #    print("t:", t_DPS[i])
    #    print("===================== DPB ===================")
    #    time_DPB = time.time()
    #    for j in range(M):
    #        u_DPB[i], t_DPB[i] = DP_bisection(noisy_array[i,:,:],f2py_cp_denoise, noise_std = sig, t_left = 0.8)
    #    print("TIME:", (time.time() - time_DPB)/M)
    #    print("PSNR:", psnr(u_DPB[i],u))
    #    print("|t-t_opt|:", np.abs(t_DPB[i] - t_opt))
    #    print("t:", t_DPB[i])

    #    print("===================== QOC ===================")
    #    time_QOC = time.time()
    #    for j in range(M):
    #        u_QOC[i], t_QOC[i] = QOC(noisy_array[i,:,:],f2py_cp_denoise, lam_init = 0.2/0.8, q = 0.9)
    #    print("TIME:", (time.time() - time_QOC)/M)
    #    print("PSNR:", psnr(u_QOC[i],u))
    #    print("|t-t_opt|:", np.abs(t_QOC[i] - t_opt))
    #    print("t:", t_QOC[i])
    #    print("===================== LC ===================")
    #    time_LC = time.time()
    #    for j in range(M):
    #        u_LC[i], t_LC[i] = LC_golden(noisy_array[i,:,:],f2py_cp_denoise, t_left = 0.8)
    #    print("TIME:", (time.time() - time_QOC)/M)
    #    print("PSNR:", psnr(u_LC[i],u))
    #    print("|t-t_opt|:", np.abs(t_LC[i] - t_opt))
    #    print("t:", t_LC[i])
    for i, sig in enumerate(sigmas):
        print("sigma = ", sig)
        noisy_array[i,:,:] = add_gaussian_noise(u, sig)
        print("===================== NOISY ===================")
        print("PNSR:", format(psnr(noisy_array[i,:,:], u),'.2f')) 

        u_opt, t_opt = optTV(noisy_array[i,:,:], u , f2py_quadratic_denoise, t_left = 0.0)
        print("===================== OPT ===================")
        print("PNSR:", format(psnr(u_opt, u),'.2f')) 
        print("t:", format(t_opt, '.4f'))

        print("===================== DPS ===================")
        time_DP = time.time()
        for j in range(M):
            u_DPS[i], t_DPS[i] = DP_secant(noisy_array[i,:,:],f2py_quadratic_denoise, noise_std = sig, t_0 = 1.0, t_1 = 0.99)
        print("TIME:", format((time.time() - time_DP)/M, '.2f'))
        print("PSNR:", format(psnr(u_DPS[i],u),'.2f'))
        print("|t - t_opt|:", format(np.abs(t_DPS[i] - t_opt),'.4f'))
        print("t:", format(t_DPS[i], '.4f'))
        #print("===================== DPB ===================")
        #time_DPB = time.time()
        #for j in range(M):
        #    u_DPB[i], t_DPB[i] = DP_bisection(noisy_array[i,:,:],f2py_quadratic_denoise, noise_std = sig, t_left = 0.0)
        #print("TIME:", (time.time() - time_DPB)/M)
        #print("PSNR:", psnr(u_DPB[i],u))
        #print("|t-t_opt|:", np.abs(t_DPB[i] - t_opt))
        #print("t:", t_DPB[i])

        print("===================== QOC ===================")
        time_QOC = time.time()
        for j in range(M):
            u_QOC[i], t_QOC[i] = QOC(noisy_array[i,:,:],f2py_quadratic_denoise, lam_init = 50.0, q = 0.9)
        print("TIME:", format((time.time() - time_QOC)/M,'.2f'))
        print("PSNR:", format(psnr(u_QOC[i],u),'.2f'))
        print("|t-t_opt|:", format(np.abs(t_QOC[i] - t_opt),'.4f'))
        print("t:", format(t_QOC[i],'.4f'))
        print("===================== LC ===================")
        time_LC = time.time()
        for j in range(M):
            u_LC[i], t_LC[i] = LC_golden(noisy_array[i,:,:],f2py_quadratic_denoise, t_left = 0.2,name='QV')
        print("TIME:", format((time.time() - time_QOC)/M,'.2f'))
        print("PSNR:", format(psnr(u_LC[i],u),'.2f'))
        print("|t-t_opt|:", format(np.abs(t_LC[i] - t_opt),'.4f'))
        print("t:", format(t_LC[i],'.4f'))
        print("===================== GCV ===================")
        time_GCV = time.time()
        for j in range(M):
            u_GCV[i], t_GCV[i] = GCV_golden(noisy_array[i,:,:],f2py_quadratic_denoise, t_left = 0.2)
        print("TIME:", format((time.time() - time_GCV)/M,'.2f'))
        print("PSNR:", format(psnr(u_GCV[i],u),'.2f'))
        print("|t-t_opt|:", format(np.abs(t_GCV[i] - t_opt),'.4f'))
        print("t:", format(t_GCV[i],'.4f'))

def plotdownsampled():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    #u = np.array(imageio.imread('images/graydata/5.gif')/(255.0), order = 'F')
    sigma = 0.05
    f = add_gaussian_noise(u,sigma)
    n = f.shape[0]
    u_opt, t_opt = optTV(f, u, f2py_cp_denoise, t_left = 0.8)
    #print(psnr(u_opt, u))
    print(t_opt)
    avg_img = createEstimator(f, n//2, 2, False, False, plot = True) 
    N = 200
    t_list = np.linspace(0.0,1.0,N)[1:]
    t, R_true = gridSearch(f,u,N,f2py_cp_denoise, plot = False)
    t, R_hat = gridSearch(f,avg_img,N,f2py_cp_denoise, plot = False)
    plt.plot(t_list, R_true, label = 'True loss')
    plt.plot(t_list, R_hat, label = 'Approximate loss')
    plt.grid()
    plt.xlabel(r'$t$')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    new_DP = lambda f, method : DP_secant(f,method, noise_std = sigma/2)

    TIME = time.time()
    #avg_img = createEstimator(f, n//2, 1, f2py_cp_denoise, new_DP, plot = True) 
    avg_img = createEstimator(f, n//2, 2, f2py_cp_denoise, new_DP, plot = True) 
    #avg_img = createEstimator(f, n//2, 1, f2py_cp_denoise, QOC, plot = True) 
    #avg_img = createEstimator(f, n//2, 2, f2py_cp_denoise, QOC, plot = True) 
    #avg_img = createEstimator(f, n//2, 2, f2py_cp_denoise, LC_golden, plot = True) 
    print("TIME:", time.time() - TIME)
    #avg_img = createEstimator(f, n//2, 1, f2py_quadratic_denoise, GCV_golden, plot = True) 
    plt.imsave('lenna_avg_denoise.png', avg_img, cmap = 'gray')
    u_rec, t_rec = optTV(f, avg_img, f2py_cp_denoise, t_left = 0.8)
    print(psnr(avg_img,u))
    plt.imsave('lenna_avg_denoise_TV.png', u_rec, cmap = 'gray')
    print(t_rec)
    print(psnr(u_rec, u))
    return 0

def deblurringexperiment():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    n = u.shape[0]
    m = 15

    shape = (n + m - 1, n + m - 1)
    # GAUSSIAN
    sig_blur = 1.5
    k = np.outer(signal.gaussian(m, 1*sig_blur), signal.gaussian(m, 1*sig_blur))


    k = k/np.sum(k)

    u_pad = edge_pad_and_shift(u,m)
    
    N = 50

    sigma = 0.02
    fker = rfftn(k, shape)
    fker_star = rfftn(np.flip(k), shape)
    f = center(irfftn(rfftn(u_pad, shape)*fker),n,m)
    f = add_gaussian_noise(f, sigma)
    plt.imsave('lenna_deconvolve.png', f, cmap = 'gray')
    u_TVopt, t = optTV_convolution(f, u, fker ,fker_star, shape, ChambollePock_convolution_edge, t_left = 0.99)
    plt.imsave('lenna_deconvolve_opt.png', u_TVopt, cmap = 'gray')
    print("OPTIMAL TV PARAMETER", t)
    print("OPTIMAL TV PSNR",psnr(u_TVopt, u))

    u_TVopt, t = optTV_convolution(f, u, fker ,fker_star, shape, quadraticRegularizer_convolution, t_left = 0.90, t_right = 1.0)
    print("OPTIMAL QV PARAMETER", t)
    print("OPTIMAL QV PSNR", psnr(u_TVopt, u))
    #plt.imshow(u_TVopt)
    #plt.show()
    #u_QV, t_QV = GCV_trace_convolution(f,quadraticRegularizer_convolution, fker, fker_star, shape, lam_init = (1- 0.95)/0.95)
    u_TV, t_TV = DP_secant_convolution(f,ChambollePock_convolution_edge, fker, fker_star, shape, noise_std = sigma, t_0 = 1.0, t_1 = 0.999)
    plt.imsave('lenna_deconvolve_DP.png', u_TV, cmap = 'gray')
    print("PSNR:", format(psnr(u_TV,u),'.2f'))
    print("|t - t_opt|:", format(np.abs(t_TV - t),'.4f'))
    print("t:", format(t_TV, '.4f'))

    
    #u_QV, t_QV = DP_secant_convolution(f,quadraticRegularizer_convolution, fker, fker_star, shape, noise_std = sigma, t_0 = 1.0, t_1 = 0.999)

    #print("PSNR:", format(psnr(u_QV,u),'.2f'))
    #print("|t - t_opt|:", format(np.abs(t_QV - t),'.4f'))
    #print("t:", format(t_QV, '.4f'))

    #u_TV, t_TV = LC_golden_convolution(f,ChambollePock_convolution_edge, fker, fker_star, shape, t_left = 0.50, name = 'TV')
    #u_QV, t_QV = LC_golden_convolution(f,quadraticRegularizer_convolution, fker, fker_star, shape, t_left = 0.50, name = 'QV')
    #print("PSNR:", format(psnr(u_TVQV,u),'.2f'))
    #print("|t - t_opt|:", format(np.abs(t_QV - t),'.4f'))
    #print("t:", format(t_TVQV, '.4f'))

    #u_TV, t_TV = QOC_convolution(f,ChambollePock_convolution_edge, fker, fker_star, shape, lam_init = (1- 0.999)/0.999, q = 0.95)
    u_QV, t_QV = QOC_convolution(f,quadraticRegularizer_convolution, fker, fker_star, shape, lam_init = 0.1, q = 0.95)
    u_TVQV, t = optTV_convolution(f, u_QV, fker ,fker_star, shape, ChambollePock_convolution_edge, t_left = 0.99)
    print("PSNR:", format(psnr(u_TVQV,u),'.2f'))
    print("t:", format(t, '.4f'))

    u_QV, t_QV = GCV_golden_convolution(f,quadraticRegularizer_convolution, fker, fker_star, shape, t_left = 0.90)
    u_TVQV, t = optTV_convolution(f, u_QV, fker ,fker_star, shape, ChambollePock_convolution_edge, t_left = 0.99)
    print("PSNR:", format(psnr(u_TVQV,u),'.2f'))
    print("t:", format(t, '.4f'))

    #print("PSNR:", format(psnr(u_QV,u),'.2f'))
    #print("|t - t_opt|:", format(np.abs(t_QV - t),'.4f'))
    #print("t:", format(t_QV, '.4f'))


    plt.imsave('lenna_deconvolve_QVQOC.png', u_QV, cmap = 'gray')
    #t_TV = 0
    #print("DP t", t_TV, t_QV)
    #print("NOISY", psnr(u,f))
    #print("DP PSNR",psnr(u_TV,u))
    #print("GCV PSNR",psnr(u_QV,u))
    #print("CONDITIION NUMBER: ", np.max(np.abs(fker))/np.min(np.abs(fker)))

    #plt.imshow(u_TV)
    #plt.show()
    #plt.imshow(u_QV)
    #plt.show()
    u_TVQV, t = optTV_convolution(f, u_QV, fker ,fker_star, shape, ChambollePock_convolution_edge, t_left = 0.99)

    plt.imsave('lenna_deconvolve_QVQOCTV.png', u_TVQV, cmap = 'gray')
    print(psnr(u_TVQV,u))
   
def QVTVdenoising():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    n = u.shape[0]
    sigmas = [0.05, 0.1]
    noisy_array = np.zeros((len(sigmas), n,n))
    N = 200
    t_list = np.linspace(0.0,1.0,N)[1:]
    colors = ['blue','blue','orange','orange']
    styles = ['-','--', '-','--'] 
    labels = ['True loss, ' + r'$\sigma = 0.05$', 
        'Approximate loss, ' + r'$\sigma = 0.05$',
        'True loss, ' + r'$\sigma = 0.1$',
        'Approximate loss, ' + r'$\sigma = 0.1$']
    for i, sig in enumerate(sigmas):
        noisy_array[i,:,:] = add_gaussian_noise(u,sig)
        u_QV, t_QV = optTV(noisy_array[i,:,:], u , f2py_quadratic_denoise, t_left = 0.0)
        _, R_true = gridSearch(noisy_array[i,:,:], u, N, f2py_cp_denoise, plot=False)
        _, R_QV = gridSearch(noisy_array[i,:,:], u_QV, N, f2py_cp_denoise, plot=False)
        plt.plot(t_list,R_true, color=colors[2*i],linestyle=styles[2*i], label = labels[2*i])
        plt.plot(t_list,R_QV, color=colors[2*i+1],linestyle=styles[2*i+1], label = labels[2*i+1])
    plt.grid()
    plt.xlabel(r'$t$')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def TVandQVexamples():
    u = np.array(imageio.imread('images/lenna.png', pilmode = 'F')/(255.0), order = 'F')
    sigma = 0.05
    f = add_gaussian_noise(u,sigma)
    sigmas = [0.05, 0.075, 0.1]
    for sig in sigmas:
        f = add_gaussian_noise(u,sig)
        plt.imsave('lenna' + str(sig) + '.png', f, cmap = 'gray')
    exit()
    lam_list_TV = [0.01, 0.05, 0.1, 0.5]
    lam_list_QV = [1.0, 5.0, 10.0, 50.0]
    for lam in lam_list_TV:
        u_TV = f2py_cp_denoise(f, lam, tol = 1.0e-16)
        plt.imsave('lenna_TV' + str(lam) + '.png', u_TV, cmap = 'gray')
    for lam in lam_list_QV:
        u_QV = f2py_quadratic_denoise(f, lam, tol = 1.0e-16)
        plt.imsave('lenna_QV' + str(lam) + '.png', u_QV, cmap = 'gray')
