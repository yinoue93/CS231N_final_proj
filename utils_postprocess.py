import numpy as np
import tensorflow as tf

import os

from constants import *
from utils_misc import *

import re
from PIL import Image
from scipy.misc import imread,imsave,toimage,imresize

def GT2illustration(dirName, outdir):
    fnames = [os.path.join(dirName, fname) for fname in os.listdir(dirName)]
    gtNames = []
    overlayNames = []

    for fname in fnames:
        reMatch = re.search('input[0-9]+.jpg', fname)
        if reMatch!=None:
            overlayNames.append(fname)
            gtNames.append(fname.replace('input', 'gt'))

    for gName,oName in zip(gtNames,overlayNames):
        gtImg = imread(gName).astype(float)
        overlayImg = imread(oName).astype(float)[:,:,np.newaxis]

        gtImg = imresize(gtImg, size=[int(IMG_DIM/4), int(IMG_DIM/4)])
        gtImg = imresize(gtImg, size=[int(IMG_DIM), int(IMG_DIM)])

        img = gtImg * overlayImg/255.0

        toimage(img, cmin=0, cmax=255).save(os.path.join(outdir,gName[gName.rfind('\\')+1:]))


def fnames2Tile(fnames, shape, outName):
    if len(shape)==3:
        VER_IMG_COUNT,HOR_IMG_COUNT,NUM_CHANNELS = shape
        TILE_DIM = (VER_IMG_COUNT*IMG_DIM, HOR_IMG_COUNT*IMG_DIM)
        tileImg = np.zeros(TILE_DIM+(NUM_CHANNELS,))
    else:
        VER_IMG_COUNT,HOR_IMG_COUNT = shape
        TILE_DIM = (VER_IMG_COUNT*IMG_DIM, HOR_IMG_COUNT*IMG_DIM)
        tileImg = np.zeros(TILE_DIM)
    
    count = 0
    for y in range(shape[0]):
        y_start = IMG_DIM*y
        y_end = IMG_DIM*(y+1)
        for x in range(shape[1]):
            x_start = IMG_DIM*x
            x_end = IMG_DIM*(x+1)

            tileImg[y_start:y_end, x_start:x_end] = imread(fnames[count]).astype(float)

            count += 1
            if count==len(fnames):
                break
        if count==len(fnames):
            break

    toimage(tileImg, cmin=0, cmax=255).save(outName)

def getOutImgNames(inDir, ids=[]):
    fnames = [os.path.join(inDir, fname) for fname in os.listdir(inDir)]

    gtNames,inNames,ovNames,preNames = [],[],[],[]

    for fname in fnames:
        if ids!=[]:
            if int(re.search('[0-9]+', re.search('[0-9]+.jpg', fname).group(0)).group(0)) not in ids:
                continue

        if re.search('gt[0-9]+.jpg', fname)!=None:
            gtNames.append(fname)
        elif re.search('input[0-9]+.jpg', fname)!=None:
            inNames.append(fname)
        elif re.search('overlay[0-9]+.jpg', fname)!=None:
            ovNames.append(fname)
        elif re.search('predicted[0-9]+.jpg', fname)!=None:
            preNames.append(fname)

    return gtNames,inNames,ovNames,preNames


def modelOutputs2Tile(inDir, outDir, outPrefix, shape, ids=[]):
    gtNames,inNames,ovNames,preNames = getOutImgNames(inDir, ids)

    fnames2Tile(gtNames, shape+(3,), os.path.join(outDir, outPrefix+'_gt.jpg'))
    fnames2Tile(inNames, shape, os.path.join(outDir, outPrefix+'_input.jpg'))
    fnames2Tile(ovNames, shape+(3,), os.path.join(outDir, outPrefix+'_overlay.jpg'))
    fnames2Tile(preNames, shape+(3,), os.path.join(outDir, outPrefix+'_predicted.jpg'))

def imgStats(inDir):
    # gtNames,inNames,ovNames,preNames = getOutImgNames(inDir)
    gtNames = ['../results/imgs/all_samples/zhang_class_lch/1239580_0_gt114.jpg']
    preNames = ['../results/imgs/all_samples/zhang_class_lch/1239580_0_predicted114.jpg']
    num_imgs = len(gtNames)

    gt_stats = [0,0,0]
    pred_stats = [0,0,0]

    for i in range(num_imgs):
        if i%10==0:
            print('.')
        gtImg = rgb2lch(imread(gtNames[i]).astype(float))
        predImg = rgb2lch(imread(preNames[i]).astype(float))

        gtImg = gtImg[180:,180:,:]
        predImg = predImg[180:,180:,:]

        for c in range(3):
            gt_stats[c] += np.mean(gtImg[:,:,c])
            pred_stats[c] += np.mean(predImg[:,:,c])

    print([val/num_imgs for val in gt_stats])
    print([val/num_imgs for val in pred_stats])

#----------------------------Weights visualizations-------------------------------

import matplotlib.pyplot as plt
from math import sqrt, ceil

def printVars():
    tfvar = tf.global_variables()
    for var in tfvar:
        print(var.name + ' : ' + str(var.get_shape().as_list()))

def getVar(sess, varName):
    tfvar = tf.global_variables()
    for var in tfvar:
        if varName in var.name:
            break
    varVal = sess.run(var)

    return varVal

def show_weights(weights, names=None):
    """
    Visualize the learned weights
    More suitable for many weights
    @type   weights :   numpy array
    @param  weights :   learned weights of shape (N,H,W,C)
    @type   names   :   list of strings
    @param  names   :   names of the weights 
    """
    plt.imshow(visualize_grid(weights, padding=1).astype('uint8'), cmap='Greys')
    plt.gca().axis('off')
    plt.show()
    plt.savefig('vis.png')

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs:
    - Xs: Data of shape (H, W, C, N)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    pixel_sz = 2
    (H, W, C, N) = Xs.shape

    Xs_resize = np.zeros((H*pixel_sz, W*pixel_sz, C, N))
    Xs = (ubound*(Xs-np.min(Xs))/(np.max(Xs)-np.min(Xs))).astype('uint8')

    for c in range(C):
        for n in range(N):
            Xs_resize[:,:,c,n] = imresize(Xs[:,:,c,n], 200, interp='nearest')
    Xs = Xs_resize

    (H, W, C, N) = Xs.shape
    low, high = np.min(Xs), np.max(Xs)

    if C==1 or C==3:
        grid_size_H = int(ceil(sqrt(N)))
        grid_size_W = int(ceil(sqrt(N)))
    else:
        grid_size_H = N
        grid_size_W = C

    count = 0
    grid_height = H * grid_size_H + padding * (grid_size_H-1)
    grid_width = W * grid_size_W + padding * (grid_size_W-1)
    grid = np.zeros((grid_height, grid_width, C))
    y0, y1 = 0, H
    for y in range(grid_size_H):
        x0, x1 = 0, W
        for x in range(grid_size_W):
            if C==1 or C==3:
                img = Xs[:,:,:,count]
                count += 1
            else:
                img = np.expand_dims(Xs[:,:,x,y], axis=-1)

            grid[y0:y1, x0:x1, :] = ubound * (img - low) / (high - low)
            x0 += W + padding
            x1 += W + padding

        y0 += H + padding
        y1 += H + padding

    if C!=3:
        grid = grid[:,:,0]
    return grid

#----------------------------End weights visualizations-------------------------------

        
if __name__ == "__main__":
    # ids = [32, 74, 81, 94, 131, 196, 237, 249]
    # ids = [12, 28, 24, 32, 74, 81, 82, 94, 116, 146, 195, 197, 213, 223, 249, 245, 237, 109]
    # ids = range(250)
    # modelOutputs2Tile('../results/imgs/all_samples/unet', '../images/img_tiles_large', 'unet', shape=(16,16), ids=ids)
    # modelOutputs2Tile('../results/imgs/all_samples/zhang_no_class', '../images/img_tiles_large', 'zhang_no_cl', shape=(16,16), ids=ids)
    # modelOutputs2Tile('../results/imgs/all_samples/zhang_class', '../images/img_tiles_large', 'zhang_cl', shape=(16,16), ids=ids)
    # modelOutputs2Tile('../results/imgs/all_samples/zhang_class_lch', '../images/img_tiles_large', 'zhang_cl_lch', shape=(16,16), ids=ids)

    # GT2illustration('../results/imgs/all_samples/zhang_no_class', '../results/imgs/all_samples/gt')

    # gtNames,_,_,_ = getOutImgNames('../results/imgs/all_samples/gt', ids)
    # fnames2Tile(gtNames, (16,16,3), '../images/img_tiles_large/gt_overlay.jpg')

    # imgStats('../results/imgs/all_samples/zhang_class_lch')

    pass