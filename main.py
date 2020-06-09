from utils import *
from ChambollePock import *
from projectedGradient import *
from quadraticTV import *
import time
from PIL import Image
import os
import numpy.random
from comparison_methodsTV import *
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
#from interpolation import *
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, rfftn, irfftn
from scipy.fftpack import dct, idct
from scipy.signal import fftconvolve, convolve, gaussian

def createDownsampled(u,n,blur = False, sigma_blur = 0.5):
    """
    Creates n downsampled images from u based on a fixed grid
    If blur, applies gaussian filter with parameter sigma_blur
    before downsampling.
    This is done to avoid aliasing
    """
    shape = u.shape
    if blur:
        f = scipy.ndimage.gaussian_filter(u, sigma = sigma_blur)
    else:
        f = np.copy(u)

    l = int(np.sqrt(n))
    m = shape[0]//n * l
    image_array = np.zeros((n,m,m))
    shift_x = 0
    shift_y = 0
    for i in range(n):
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = f[shift_x + j*l, shift_y + k*l]
        if ((shift_x+1)%l == 0 and shift_x != 0):
            shift_x =0
            shift_y +=1
        else:
            shift_x += 1
    return image_array

def createEstimator_recursive(f, n,depth, curr_depth, blur = False, sigma_blur = 0.5):
    """
    Creates empirical estimator given recursive method
    Must be called with depth = curr_depth
    """
    if curr_depth == 0:
        image_array = createDownsampled(f,n, blur = blur, sigma_blur = sigma_blur)
        image_array_upsampled = createUpsampled(image_array, f.shape)
        #fig = plt.figure(figsize=(2,2))
        #for i in range(n):
        #    fig.add_subplot(2,2,i+1)
        #    plt.imshow(image_array_upsampled[i], cmap = "gray")
        #plt.show()

        return np.average(image_array_upsampled, axis=0)
    else:
        image_array = createDownsampled(f,n)
        for i in range(n):
            image_array[i] = createEstimator_recursive(image_array[i], n, depth, curr_depth-1, blur = blur, sigma_blur = sigma_blur)
        image_array_upsampled = createUpsampled(image_array, f.shape)
        return np.average(image_array_upsampled, axis=0)

def createDownsampled_random(u,n,N,blur = False, sigma_blur = 0.5):
    """
    Creates downsampled with random method, similar to createDownsampled
    """
    shape = u.shape
    if blur:
        f = scipy.ndimage.gaussian_filter(u, sigma = sigma_blur)
    else:
        f = np.copy(u)
    l = int(np.sqrt(n))
    m = shape[0]//n * l
    newshape = (m,m)
    image_array = np.zeros((N,m,m))
    cord_array = np.arange(0,l)
    for i in range(N):
        x_cords = np.random.choice(cord_array, m*m)
        y_cords = np.random.choice(cord_array, m*m)
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = f[x_cords[j*m + k] + j*l,y_cords[j*m + k] + k*l]
    return image_array

def createResampledArray(u, n, N):
    """
    n is the one side of the square you are extending. Choose this to be odd for now
    """
    m = u.shape[0]
    pad_n = (n-1)//2 # DEAL WITH EVEN n LATER
    print(pad_n)
    image_array = np.zeros((N,m,m))
    u_padded = np.pad(u, (pad_n,pad_n), mode = 'edge')
    
    cord_array = np.arange(0,n)
    for i in range(N):
        x_cords = np.random.choice(cord_array, m*m)
        y_cords = np.random.choice(cord_array, m*m)
        for j in range(m):
            for k in range(m):
                image_array[i,j,k] = u_padded[x_cords[j*m + k] + j,y_cords[j*m + k] + k]
    return image_array

def ssimdplots():
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sigma = 0.1
    f = add_gaussian_noise(u,sigma)
    N = 50
    t_list = np.linspace(0.5, 0.99, N)

    ssim_lists = np.zeros((4, N))
    win_sizes = [31, 15, 7]
    for i in range(0,N):
        t = t_list[i]
        u_t = ChambollePock_denoise(f, (1-t)/t, tau = 0.4, sig = 0.25, acc = True, tol = 1e-4)
        for j, win_size in enumerate(win_sizes):
            ssim_lists[j,i] = ssim(u_t, u, win_size = win_size)
    for i, win_size in enumerate(win_sizes):
        plt.plot(ssim_lists[i], label = win_size)
        plt.legend()
    plt.show()
    
def gaussianplots():
    u = imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    m = u.shape[0]
    sig = 0.05
    f = add_gaussian_noise(u, sigma = sig)
    N = 100
    sigs = np.linspace(0,2,N)
    #u_blur = np.zeros((N,m,m))
    psnrs = np.zeros(N)
    for i,sig_blur in enumerate(sigs):
        u_blur = scipy.ndimage.gaussian_filter(f,sig_blur)
        psnrs[i] = psnr(u,u_blur)
    plt.plot(sigs,psnrs)
    plt.show()

def createUpsampled(image_array, orig_shape):
    """
    Creates upsampled image
    TODO, use another library for resizing,
    like scipy or scikit
    """
    shape = image_array.shape
    image_array_upsampled = np.zeros((shape[0], ) + orig_shape)
    for i in range(shape[0]):
        image = Image.fromarray(image_array[i])
        image = image.resize(orig_shape, Image.BILINEAR)
        image_array_upsampled[i] = np.array(image)
    return image_array_upsampled

def gram_schmidt_vec(image_array):
    shape = image_array.shape
    image_array_vec = image_array.reshape((shape[0], shape[1]*shape[2]))
    basis_img = np.zeros((shape[0],shape[1]*shape[2]))
    next_img = image_array_vec[0]/np.linalg.norm(image_array[0])
    basis_img[0] = next_img
    for i in range(1, shape[0]):
        proj_sum = np.zeros((shape[1]*shape[2]))
        next_img = image_array_vec[i]
        for j in range(i):
            proj_sum += np.inner(next_img, basis_img[j]) * next_img
        next_img = next_img - proj_sum 
        basis_img[i] = next_img/np.linalg.norm(next_img)
    return basis_img

def gram_schmidt(image_array):
    shape = image_array.shape
    basis_img = np.zeros(shape)
    next_img = image_array[0]/np.linalg.norm(image_array[0], 'fro')
    basis_img[0] = next_img
    for i in range(1, shape[0]):
        proj_sum = np.zeros((shape[1],shape[2]))
        next_img = image_array[i]
        for j in range(i):
            proj_sum += np.linalg.norm(np.multiply(next_img, basis_img[j]), 'fro') * basis_img[j]
        next_img = next_img - proj_sum 
        basis_img[i] = next_img/np.linalg.norm(next_img, 'fro')
    return basis_img

def modified_gram_schmidt(image_array):
    shape = image_array.shape
    basis_img = np.zeros(shape)
    basis_img_curr = np.zeros((shape[1],shape[2]))
    for i in range(shape[0]):
        basis_img_curr = image_array[i]
        for j in range(i):
            basis_img_curr = basis_img_curr - np.sum(np.multiply(basis_img[j], basis_img_curr)) * basis_img[j]
        basis_img[i] = basis_img_curr/np.linalg.norm(basis_img_curr, 'fro')
    return basis_img

def modified_gram_schmidt_vec(image_array):
    shape = image_array.shape
    image_array_vec = image_array.reshape((shape[0], shape[1]*shape[2]))
    basis_img = np.zeros((shape[0],shape[1]*shape[2]))
    basis_img_curr = np.zeros(shape[1]*shape[2])
    for i in range(shape[0]):
        basis_img_curr = image_array_vec[i]
        for j in range(i):
            basis_img_curr = basis_img_curr - np.inner(basis_img[j], basis_img_curr)* basis_img[j]
        basis_img[i] = basis_img_curr/np.linalg.norm(basis_img_curr)
    return basis_img

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
def dct2(f,n,type=1, norm='ortho'):
    #temp = np.pad(f, (n//2,n//2))
    return dct(dct(f.T, n=2*n,type = type, norm = norm).T, n=2*n, type = type, norm = norm)

def idct2(f,n,type=1, norm='ortho'):
    return idct(idct(f.T, n=2*n,type = type, norm = norm).T, n=2*n,type = type, norm = norm)
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

def showFourier(f):
    return fftshift(np.log(np.abs(f)))

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
    m = 63

    shape = (n + m - 1, n + m - 1)
    # GAUSSIAN
    sig_blur = 0.5
    k = np.outer(signal.gaussian(m, 2*sig_blur), signal.gaussian(m, 1*sig_blur))

    # MEAN
    #k = np.ones((m,m))

    # MOTION BLUR
    #k = np.zeros((m,m))
    #k[m//2,:] = 1.0
    k = k/np.sum(k)

    # MOTION BLUR WITH "PROPER" PADDING
    #blur_range = 3
    #k = np.zeros((m,m))
    #k[m//2,(m//2-blur_range):-(-blur_range+m//2)] = 1.0
    #k = k/np.sum(k)
    #sigma = 0.01
    u_pad = edge_pad_and_shift(u,m)
    k_padded = np.pad(k,(0,n), 'constant')


    fker = rfftn(k_padded,shape)
    fker_star = rfftn(np.flip(k), shape)

    
    sigma = 0.02
    #f = center(add_gaussian_noise(irfftn(rfftn(u_pad, shape)*fker),sigma),n,m)
    #f = center(add_gaussian_noise(irfftn(rfftn(u, shape)*fker),sigma),n,m)
    #f = add_gaussian_noise(irfftn(rfftn(u_pad, shape)*fker),sigma)
    f = add_gaussian_noise(center(irfftn(rfftn(u_pad, shape)*fker),n,m), sigma)
    #f[100,100] +=0.01 
    plt.imshow(f, cmap = 'gray')
    plt.show()

    #naive_deconvolve = irfftn(rfftn(edge_pad_and_shift(f, m), shape)/fker)

    naive_deconvolve = irfftn(rfftn(np.pad(f, ((m-1)//2, (m-1)//2), 'edge'), shape)/fker)
    #naive_deconvolve = irfftn(rfftn(f)/fker)
    ##plt.imshow(edge_pad_and_shift(f, m))
    ##plt.show()
    ##naive_deconvolve = irfftn(rfftn(np.pad(f,((m-1)//2,(m-1)//2)),shape)/fker)
    print(psnr(naive_deconvolve[:n,:n],u))
    plt.imshow(naive_deconvolve[:n,:n], cmap = 'gray')
    plt.show()

    print("CONDITIION NUMBER: ", np.max(np.abs(fker))/np.min(np.abs(fker)))
    #fker_star = rfftn(k,shape)
    u_opt,t_opt = optTV_golden_convolution(f,u,fker,fker_star,shape,tol = 1e-4)
    #t_opt, R_list = gridSearch_convolution(f,u,100,fker,fker_star,shape)
    #plt.plot(R_list)
    #plt.show()
    print(t_opt)
    result = ChambollePock_convolution_edge(f,(1-t_opt)/t_opt, fker, fker_star,shape)
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

    exit()
    u =  imageio.imread('images/lenna.png', pilmode = 'F')/255.0
    sigma = 0.3
    f = add_gaussian_noise(u,sigma)
    plt.imsave("lenna_noisy.png", f, cmap = 'gray')
    orig_shape = f.shape
    n = 4
    n_rec = 4
    nn = np.ceil(np.sqrt(n))

    sigma_blur = 0.0
    print("Original PSNR:", psnr(u,f))
    avg_img_recursive = createEstimator_recursive(f,n = n_rec, depth = 1, curr_depth =1)#, blur = True, sigma_blur = sigma_blur/4)
    plt.imshow(avg_img_recursive, cmap = 'gray')
    plt.show()
    print(psnr(avg_img_recursive,u))
    #image_array_fixed = createDownsampled_recursive(f,n, depth = 1, curr_depth = 1, blur = True, sigma_blur = sigma_blur)
    #image_array_upsampled_fixed = createUpsampled(image_array_fixed,orig_shape)

    #avg_img_fixed = np.average(image_array_upsampled_fixed, axis=0)

    plt.imsave("lenna_avg.png", avg_img_recursive, cmap = 'gray')
    tol = 1e-4 
    u_opt, t_opt = optTV_golden(f,u, tol = tol)
    plt.imsave("lenna_opt.png", u_opt, cmap = 'gray')
    u_opt_fixed, t_opt_fixed = optTV_golden(f,avg_img_recursive, tol = tol)
    plt.imsave("lenna_opt_fixed.png", u_opt_fixed, cmap = 'gray')

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

def ringingFunc(x,y):
    if (x**2 + y**2 < 0.77):
        return np.exp(x**2 + y**2)
    else:
        return np.exp(x**2 + y**2)/2

def forPresentation():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/255.0
    plt.imsave('lenna_laplace.png',scipy.ndimage.laplace(u), cmap = 'gray')
    sigma = 0.05
    f = add_gaussian_noise(u,sigma)
    """
    GRID SEARCHES PLOT
    """
    N = 100
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
    blurred = scipy.ndimage.gaussian_filter(f,1.0)

    print("GAUSSIAN BLUR PSNR:",psnr(blurred,u))

    #t_opt, R_list = gridSearch(f,u,N)
    #t_opt, R_list_blurred = gridSearch(f,blurred,N)
    #plt.plot(R_list, label =r'True: $\|u^\dagger - u^t\|$' )
    #plt.plot(R_list_blurred, label =r'Approximate: $\|\hat{u} - u^t\|$' )
    #plt.grid()
    #plt.legend()
    #plt.title(r'Loss function for true and approximate data')
    #plt.xlabel(r'$t$')
    #plt.ylabel(r'Loss')
    #plt.show()
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

def ringingExample():
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

def TV_testing():
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

def backInIt():
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

def GCV_testing():
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
    
def quadratictesting():
    u =  imageio.imread('images/lenna256.jpg', pilmode = 'F')/(3*255.0)
    #u[:,:] = 0.0
    n = u.shape[0]

    #u[:n//2,:] = 0.5
    #for i in range(n//2, n):
    #    for j in range(0,n,2):
    #        u[i,i%2 + j] = 0.5

    #sig_blur = 1.0
    #m = 31
    #shape = (n + m - 1, n + m - 1)
    #k = np.outer(signal.gaussian(m, 1*sig_blur), signal.gaussian(m, 1*sig_blur))
    #k = k/np.sum(k)
    #fker = rfftn(k, shape) 
    #fker_star = rfftn(np.flip(k), shape)
    sigma = 0.1
    #f = add_gaussian_noise(center(irfftn(rfftn(edge_pad_and_shift(u,m), shape)*fker),n,m), sigma)
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
        #result = quadraticRegularizer_convolution(f,(1-t)/t, fker, fker_star, tol = 1.0e-5)
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

    #u_opt, t_opt = optTV_golden(f,u_hat, tol = 1.0e-4)
    #u_true, t_true = optTV_golden(f,u, tol = 1.0e-4)
    u_opt, t_opt = optTV_golden_convolution(f,u_hat, fker, fker_star, shape,tol = 1.0e-4)
    u_true, t_true = optTV_golden_convolution(f,u, fker, fker_star, shape,tol = 1.0e-4)

    print(psnr(u_true,u))
    print(psnr(u_opt,u))
    plt.imshow(u_opt, cmap='gray')
    plt.show()
    plt.imshow(u_true, cmap='gray')
    plt.show()

    
if __name__ == "__main__":
    np.random.seed(316)
    quadratictesting()
    #GCV_testing()
    
    #TV_testing()
    #testConvolutionDownsampling()
    #gaussianplots()
    #ssimdplots()
    #backInIt()
    #testConvolution()
    #testConvolutionOperator()
    #plotFourier()
    #plotConvolution()
    #ringingExample()
    #test_gauss_opt_sig()
    #test_gauss()
    #testInterpolation()
    #testMean()
    #testDownsampled()
    #TV_convergence()
    #testOptTV()
    #testParameterSelections()
    #testEstimator()
    #exit()
    #TV_convergence()
    #testOtherParameterSelections()
    #forPresentation()
    
