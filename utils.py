import scipy as sp
from scipy.fftpack import dct, idct
from scipy.linalg import circulant, toeplitz
import autograd.numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfftn, irfftn
from PIL import Image

def dct2(v, type = 3, s = [None,None], norm = None):
    return dct(dct(v, n = s[1], type = type, norm = norm, axis = 0),
            n = s[0], type = type, norm = norm, axis = 1)
def idct2(v, type = 3, s = [None, None], norm = None):
    return idct(idct(v, n = s[1], type = type, norm = norm, axis = 0),
            n = s[0], type = type, norm = norm, axis = 1)

def center(f,n,m):
    """
    Returns centered image of size nxn that has
    been padded to size n+m-1xn+m-1
    """
    return f[m//2:m//2+n, m//2:m//2+n]

def edge_pad_and_shift(f, m):
    """
    pads image to size n+m-1xn+m-1 with edge padding
    """
    f_pad = np.pad(f,((m-1)//2,(m-1)//2), 'edge')
    #f_pad = np.pad(f,((m-1)//2,(m-1)//2), 'constant')
    f_pad = np.roll(f_pad, -(m-1)//2, axis = 0)
    f_pad = np.roll(f_pad, -(m-1)//2, axis = 1)
    return f_pad

def add_gaussian_noise(f, sigma=0.001):
    """
    Adds gaussian noise to image
    Assumes image is rescaled to lie between 1.0 and 0.0.
    """
    out = np.zeros((2,) + f.shape, f.dtype)

    shape = f.shape
    #out = f + sigma* numpy.random.normal(0,1,np.prod(shape)).reshape(shape)
    out = np.minimum(np.maximum(f + sigma* np.random.normal(0,1,np.prod(shape)).reshape(shape), 0.0), 1.0)
    return out

def add_gaussian_blurring(f, ker, sigma):
    out = signal.fftconvolve(f,ker, mode='same')
    return out

def D_zero_boundary(u):
    """
    Calculates gradient of image u of size n,m
    returns gradient of size 2,n,m
    Gradient as given by chambolle
    """
    out = np.zeros((2,) + u.shape, u.dtype)

    # x-direction
    out[0, :-1, :] = u[1:, :] - u[:-1, :]

    # y-direction
    out[1, :, :-1] = u[:, 1:] - u[:, :-1]
    return out

def Dast_zero_boundary(f):
    """
    Calculates divergence of image f of size 2,n,m
    returns divergence of size n,m
    """
    out = np.zeros_like(f)

    # Boundaries along x-axis
    out[0, 0, :] = - f[0, 0, :]
    out[0, -1, :] = f[0, -2, :]
    # Inside along x-axis
    out[0, 1:-1, :] = f[0, :-2, :] - f[0, 1:-1, :] 

    # Boundaries along y-axis
    out[1, :, 0] = -f[1, :, 0]
    out[1, :, -1] = f[1, :, -2]
    # Inside along y-axis
    out[1, :, 1:-1] = f[1, :, :-2] - f[1, :, 1:-1]

    # Return sum
    return np.sum(out, axis=0)

def D_convolution(u):
    """
    Calculates gradient of image u of size n,m
    returns gradient of size 2,n,m
    Gradient calculated so that it corresponds
    to convolution with forward differences [0,-1,1] and [0,-1,1]^T.
    
    """
    out = np.zeros((2,) + u.shape, u.dtype)

    out[0, :-1, :] = u[1:, :] - u[:-1, :]
    out[0, -1, :] = 0 #-u[-1,:]

    out[1, :, :-1] = u[:, 1:] - u[:, :-1]
    out[1, :, -1] = 0 #-u[:,-1]

    return out

def Dast_convolution(p):
    """
    Calculates adjoint of gradient (negative divergence)
    of an "adjoint" image p of size 2,n,m
    returns adjoint of gradient of size n,m.
    Adjoint of D.
    """

    out = np.zeros_like(p)

    out[0, 1:, :] =  p[0, :-1, :] - p[0, 1:, :]
    out[0, 0, :] = 0 #-p[0, 0, :]

    out[1, :, 1:] =  p[1, :, :-1] - p[1, :, 1:]
    out[1, :, 0] = 0 #-p[1, :, 0]

    return np.sum(out, axis=0)


def norm(f, axis=0, keepdims=False):
    """
    returns euclidean norm of image f of size n,m
    """
    return np.sqrt(np.sum(f**2, axis=axis, keepdims=keepdims))

def psnr(noisy,true):
    """
    Calculates psnr between two images that are scaled between 0 and 1
    """
    MSE = np.mean((noisy-true)**2)
    #MAX = max(np.max(noisy),np.max(true))
    MAX = 1.0
    return 10 * np.log10(MAX**2/MSE)
    #return 20*np.log10(mse) - 10*np.log10(np.max(np.max(noisy),np.max(true)))

def dualitygap_denoise(lam,u,divp,f):
    #return 0.5 * lam * np.linalg.norm(u - f)**2 +  np.sum(norm1(grad(u))) + 0.5 * np.linalg.norm(f + (1/lam)*divp)**2 - 0.5 * np.linalg.norm(f)**2
    return 0.5 * lam * np.linalg.norm(u - f,'fro')**2 + np.sum(norm(grad(u))) + (1/(2*lam))*np.linalg.norm(divp,'fro')**2 + np.sum(np.multiply(f,divp))

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


def showFourier(f, log = True):
    if (log):
        return fftshift(np.log(np.abs(f)))
    else:
        return fftshift(np.abs(f))

