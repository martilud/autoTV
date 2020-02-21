import numpy as np

def upsample(f,newshape, method):
    if method == 'nn':
        return nearest(f,newshape)

def nearest(f, newshape):
    shape = f.shape
    out = np.zeros(newshape)
    rowratio, colratio = newshape[0]/shape[0], newshape[1]/shape[1]
    newrows = np.floor(np.arange(newshape[0])/rowratio).astype(int)
    newcols = np.floor(np.arange(newshape[1])/colratio).astype(int)
    print(newrows)
    for i, row in enumerate(newrows):
        for j, col in enumerate(newcols):
            out[i,j] = f[row,col]
    return out
