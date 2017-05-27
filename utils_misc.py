import random
import os
import sys
import numpy as np
import re
import h5py
import scipy
import math
import shutil

from constants import *

from shutil import copyfile
from scipy.misc import imread,imsave,toimage,imresize
from multiprocessing import Pool
from PIL import Image

def makeDir(dirname):
    # make the directory for the created files
    try:
        os.makedirs(dirname)
    except:
        pass

def printSeparator(title):
    print('\n' + '-'*25 + title + '-'*25)

def findMean(dataDir):
    m = 0
    fnames = os.listdir(dataDir+'raw_processed_reduced'+'\\')
    fnameLen = float(len(fnames))

    dirMod = ['binary','line','reduced']
    for mod in dirMod:
        avgR,avgG,avgB = 0,0,0
        for i,fname in enumerate(fnames):
            if i%int(fnameLen*0.1)==1:
                print('. R: %f, G: %f, B: %f' %(avgR*fnameLen/i, avgG*fnameLen/i, avgB*fnameLen/i))
            imgs = imread(dataDir+'raw_processed_'+mod+'\\'+fname)
            if len(imgs.shape)==3:
                avgR += np.mean(imgs[:,:,0])/fnameLen
                avgG += np.mean(imgs[:,:,1])/fnameLen
                avgB += np.mean(imgs[:,:,2])/fnameLen
            else:
                avgR += np.mean(imgs[:,:])/fnameLen

        print(mod)
        print('R: %f' % avgR)
        print('G: %f' % avgG)
        print('B: %f' % avgB)


#--------------------------H5 MODULES---------------------------------

def jpg2H5(dataPack):
    # a helper function for zipDirectory()
    inputDir, outputDir, filenames, toDir = dataPack

    with h5py.File(toDir,'w') as hf:
        for filename in filenames:
            img_input = imread(os.path.join(inputDir, filename))
            img_output = imread(os.path.join(outputDir, filename))
            img = np.dstack((img_input, img_output))

            hf.create_dataset(filename.replace('.jpg',''), 
                              data=img)

def numpy2jpg(outputFname, arr, overlay=None, meanVal=0, verbose=False):
    outputImg = arr[:,:,0] if (len(arr.shape)==3 and arr.shape[2]==1) else arr
    
    print(np.max(outputImg))
    print(np.min(outputImg))
    print(np.mean(outputImg))
    

    if meanVal!=None:
        if len(outputImg.shape)==2:
            outputImg = (outputImg + meanVal)
        else:
            outputImg = (outputImg + [REDUCED_R_MEAN,REDUCED_G_MEAN,REDUCED_B_MEAN])

    if overlay!=None:
        outputImg *= (overlay + LINE_MEAN)/255.0

    # if verbose, print out the image's mean values
    if verbose:
        print(outputFname)
        if len(outputImg.shape)==2:
            print(np.mean(outputImg))
        else:
            print(np.mean(outputImg[:,:,0]))
            print(np.mean(outputImg[:,:,1]))
            print(np.mean(outputImg[:,:,2]))

    toimage(outputImg, cmin=0, cmax=255).save(outputFname)


def gaussianDist(pt1, pt2):
    std = 0.5
    sqdist = np.sum((pt1-pt2)**2, axis=1)
    return np.exp(-sqdist/std)

def map_output(outData, outImgSz):
    import itertools
    
    # resize
    resizedData = imresize(outData, size=[outImgSz, outImgSz]).astype(int)
    
    # create 1-hot vector
    encoding_sz = outImgSz*outImgSz
    soft_encoding = np.zeros(shape=(encoding_sz,512))
    
    discr_data = resizedData.reshape([encoding_sz, -1]) / 32.0
    centered = (discr_data%1 - 0.5)
    
    discr_data_int = discr_data.astype(int)
    rval = discr_data_int[:,0]
    gval = discr_data_int[:,1] << 3
    bval = discr_data_int[:,2] << 6
    
    rs = [(rval, 0), (np.minimum(rval+1, 7), 1),  (np.maximum(rval-1, 0), -1)]
    gs = [(gval, 0), (np.minimum(gval+8, 56), 1),  (np.maximum(gval-8, 0), -1)]
    bs = [(bval, 0), (np.minimum(bval+64, 448), 1),  (np.maximum(bval-64, 0), -1)]
    
    coords = [rs, gs, bs]
    params = list(itertools.product(*coords))
    
    for (rv, roff),(gv, goff),(bv, boff) in params:    
        indx_1d = (rv + gv + bv)
        soft_encoding[range(encoding_sz), indx_1d] += gaussianDist(centered, [roff, goff, boff])
    
    # normalize, and clean up for efficient storage
    soft_encoding = soft_encoding.astype(np.float16)
    soft_encoding = soft_encoding / np.sum(soft_encoding, axis=1)[:, np.newaxis]
    soft_encoding[soft_encoding<1e-4] = 0
    soft_encoding = soft_encoding / np.sum(soft_encoding, axis=1)[:, np.newaxis]
    
#    picid = 2500
#    print(soft_encoding[picid,:])
#    print(resizedData.reshape([encoding_sz,-1])[picid])
#    print(discr_data[picid])
#    print(rval[picid], gval[picid], bval[picid])
#    print(centered[picid, 0],centered[picid, 1],centered[picid, 2])
#    indx = (rval[picid] + gval[picid] + bval[picid])
#    print(indx)
#    print(soft_encoding[picid,indx])
#    exit(0)
    
    return soft_encoding


def h52numpyWorker(dataPack):
    hdf5Filename, keys, batch_sz, mod_output = dataPack

    inData = np.empty(shape=(len(keys), IMG_DIM, IMG_DIM), dtype=float)
    out_img_shape = (len(keys), int(IMG_DIM/4)**2, 512) if mod_output else (len(keys), IMG_DIM, IMG_DIM, 3)
    outData = np.empty(shape=out_img_shape, dtype=float)
    fileNames = []

    with h5py.File(hdf5Filename,'r', driver='core') as hf:
        for i,key in enumerate(keys):
            # check the data loading speed (for debug use)
            #if i%int(len(keys)*0.1)==0:
            #    print('===============++++++++++++++++++=================')
                
            indx = i%batch_sz
            
            if mod_output:
                if '_output' in key:
                    continue
                    
                inData[i,:,:] = hf.get(key)[:]
                outData[i,:,:] = hf.get(key+'_output')[:]
            else:
                data = hf.get(key)
                inData[i,:,:] = data[:,:,0]
                outData[i,:,:,:] = data[:,:,1:]
                
            fileNames.append(key.replace('\\','/'))
            if '.jpg' not in fileNames[-1]:
                fileNames[-1] += '.jpg'
    
    return (inData, outData, fileNames)

def h52numpy(hdf5Filename, checkMean=False, batch_sz=1, mod_output=False, iter_val=None, shuffle=True):
    """
    Returns a shuffled list of input and output data, with each element of the list
    numpy array of shape (batch_sz, H, W, C)
    """

    if 'line' in hdf5Filename:
        input_mean = LINE_MEAN
    elif 'binary' in hdf5Filename:
        input_mean = BINARY_MEAN
        
    meanTotal = np.asarray([0]*4)
    count = 0

    with h5py.File(hdf5Filename,'r') as hf:
        keys = list(hf.keys())

        # if iter_val is specified, only load a portion of the data
        if iter_val!=None:
            stride_sz = int(len(keys)/4)
            keys = keys[iter_val*stride_sz:(iter_val+1)*stride_sz]

        if shuffle:
            random.shuffle(keys)
        # make the key size a multiple of batch_sz
        keys = keys[:batch_sz*int(len(keys)/batch_sz)]

    workerKeySz = int(math.ceil(len(keys)/float(POOL_WORKER_COUNT)))
    keyList = [keys[i*workerKeySz:(i+1)*workerKeySz] for i in range(POOL_WORKER_COUNT)]
    dataPack = [(hdf5Filename, k, batch_sz, mod_output) for k in keyList]

    p = Pool(POOL_WORKER_COUNT)
    results = p.map(h52numpyWorker, dataPack)

    inDataArr = []
    outDataArr = []
    fileNameArr = []
    for result in results:
        iData,oData,fN = result
        inDataArr.append(iData)
        outDataArr.append(oData)
        fileNameArr.append(fN)

    # consolidate the lists
    inData = np.concatenate(inDataArr)
    outData = np.concatenate(outDataArr)
    fileNames = sum(fileNameArr, [])

    inData = inData - input_mean
    inData = np.expand_dims(inData, axis=3)
    if not mod_output:
        outData = outData.astype(int) - [REDUCED_R_MEAN,REDUCED_G_MEAN,REDUCED_B_MEAN]

    if checkMean:
        print(count)
        print(meanTotal)
        print('Means: ' + str(meanTotal/count))

    return inData, outData, fileNames


def repackH5Worker(dataPack):
    fromName, toName, compression = dataPack
    print('Working on %s...' % fromName)

    with h5py.File(fromName,'r') as fromHF, h5py.File(toName,'w') as toHF:
        keys = list(fromHF.keys())

        for i,key in enumerate(keys):
            data = fromHF.get(key)

            inData = data[:,:,0]
            outData = map_output(data[:,:,1:].astype(int), int(IMG_DIM/4))
            
            toHF.create_dataset(key, data=inData, compression=compression)
            toHF.create_dataset(key+'_output', data=outData, compression=compression)

def repackH5(dataDir, outputDir, compression='gzip'):
    makeDir(outputDir)

    dataPacks = []
    for filename in os.listdir(dataDir):
        fromName = os.path.join(dataDir, filename)
        toName = os.path.join(outputDir, filename)
        
        if (os.path.isfile(toName)) or ('filepart' in fromName):
            print('Excluding %s, because the file already exists, or is a .filepart...' % toName)
        else:
            dataPacks.append((fromName, toName, compression))

    p = Pool(POOL_WORKER_COUNT)
    p.map(repackH5Worker, dataPacks)

#--------------------------END H5 MODULES---------------------------------


def cleanUpDatasetWorker(dataPack):
    dataDir,filename = dataPack
    outfolder = 'to_be_removed'

    imgPath = os.path.join(dataDir, filename)
    try:
        f = open(imgPath, 'rb')
        img = Image.open(f)
        img.load()
        f.close() # release the memory
    except:
        print(imgPath)
        f.close() # release the memory
        copyfile(imgPath, os.path.join(outfolder, filename))
        os.remove(imgPath)

def cleanUpDataset(dataDir):
    outfolder = 'to_be_removed'
    makeDir(outfolder)

    filenames = os.listdir(dataDir)
    filenames = [(dataDir, fname) for fname in filenames]

    p = Pool(POOL_WORKER_COUNT)
    p.map(cleanUpDatasetWorker, filenames)


#--------------------------ZIPPING/UNZIPPING MODULES---------------------------------

import zipfile
def zipper(dataPack):
    # a helper function for zipDirectory()
    fromDir, _, filenames, toDir = dataPack

    with zipfile.ZipFile(toDir, 'w', zipfile.ZIP_DEFLATED) as f:
        for filename in filenames:
            fullname = os.path.join(fromDir, filename)
            f.write(fullname, arcname=filename)

def zipDirectory(dataDir, outputDirName=None, zipFileSz=1024, originalDir=None, overwrite=True):
    # compress the directories into chunks of zip/hdf5 files

    if dataDir[-1]=='/':
        dataDir = dataDir[:-1]

    if outputDirName==None:
        outputDirName = '%s_compressed' % dataDir

    print("Zipping up %s to %s" %(dataDir,outputDirName))

    makeDir(outputDirName)

    filenames = os.listdir(dataDir)

    zipNum = 0
    currSz = 0
    zipNames = []
    dataPacks = []
    for filename in filenames:
        currSz += os.stat(os.path.join(dataDir,filename)).st_size
        zipNames.append(filename)

        if currSz > (zipFileSz<<20):
            if zipNum%10==0:
                print('.')

            fileID = zipNum if overwrite else (zipNum+len(os.listdir(outputDirName)))
            toDir = '%s/compressed_%d' % (outputDirName, fileID)

            dataPacks.append((dataDir, originalDir, zipNames, toDir))

            zipNum += 1
            currSz = 0
            zipNames = []

    if currSz != 0:
        fileID = zipNum if overwrite else (zipNum+len(os.listdir(outputDirName)))
        toDir = '%s/compressed_%d' % (outputDirName, fileID)
        dataPacks.append((dataDir, originalDir, zipNames, toDir))

    print('Processes dispatched...')

    compress_func = zipper if (originalDir==None) else jpg2H5
    p = Pool(POOL_WORKER_COUNT)
    # dataPacks = (dataPacks[0],)
    p.map(compress_func, dataPacks)


def unzipper(dataPack):
    fullname, testDir = dataPack
    zip_ref = zipfile.ZipFile(fullname, 'r')
    zip_ref.extractall(testDir)
    zip_ref.close()

def unzipDirectory(dataDir, outputDir):
    # uncompress the .zip files in the specified directory

    makeDir(outputDir)

    dataPacks = []
    for filename in os.listdir(dataDir):
        fullname = os.path.join(dataDir, filename)
        dataPacks.append((fullname, outputDir))

    p = Pool(POOL_WORKER_COUNT)
    p.map(unzipper, dataPacks)

def load_sample(dataDir):
    print('Unzipping the directory %s' % dataDir)
    unzipDirectory(dataDir)
    print('Unzipping successful....')

#--------------------------END ZIPPING/UNZIPPING MODULE---------------------------------

if __name__ == "__main__":
    # getBWPics()
    # zipDirectory('test_scraped_processed_binary', outputDirName='D:\\Backups\\CS231N_data\\processed_binary', zipFileSz=1024)
    # unzipDirectory('test_scraped_compressed', 'test_scraped_uncompressed')

    # cleanUpDataset('test_scraped')

    # zipDirectory('test_imgs', outputDirName='D:\\Backups\\CS231N_data\\processed_lines_np', 
    #            zipFileSz=1024, pic2h5=True)
    

    # zipDirectory('small_scraped_processed_line', outputDirName='tmp', 
    #            zipFileSz=1024, originalDir='small_scraped_processed_reduced')
    # h52numpy('D:\\Backups\\CS231N_data\\line\\line_dataset_26', 'tmp3')

    # unzipDirectory('D:\\Backups\\CS231N_data\\scraped', outputDir='scraped')

    # findMean('D:\\Backups\\CS231N_data\\tmp\\')
    
    # unzipper(('D:\\Backups\\CS231N_data\\scraped\\compressed_26', 'tmp4'))
    
    # repackH5('/home/tbonerocksyinoue/data/line', outputDir='/home/tbonerocksyinoue/data/line_classification', compression='lzf')
        
    pass