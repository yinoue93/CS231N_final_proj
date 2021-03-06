import numpy as np
import cv2
import os
import random
import time
import re
from multiprocessing import Pool

IMG_SZ = 256.0
BW_THRESHOLD = 20
POOL_WORKER_COUNT = 6

# much of the code borrowed from http://qiita.com/khsk/items/6cf4bae0166e4b12b942#_reference-d2b636028b7b8fbf4b34

def makeDir(dirname):
    # make the directory for the created files
    try:
        os.makedirs(dirname)
    except:
        pass

neiborhood8 = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]],
                        np.uint8)

def getBWPics():
    folderName = 'test_scraped'
    outfolder = 'bw_test'
    makeDir(outfolder)
    num_imgs = 100

    imgList = [os.path.join(folderName, filename) for filename in os.listdir(folderName)]
    random.shuffle(imgList)

    counter = 0
    for i,imgPath in enumerate(imgList):
        try:
            imgs = loadImg(imgPath)
        except:
            continue
        if not len(imgs):
            counter += 1
            print(imgPath)
            copyfile(imgPath, imgPath.replace(folderName,outfolder))

            if counter==num_imgs:
                break

    print(counter/float(i))

def loadImg(imgPath, removeBW, verbose=False):
    """
    returns an array of images that is cropped to (IMG_SZ, IMG_SZ)
    the first element of the returned array is a @returnMsg where:
    0 - all images are colored
    1 - at least 1 images are BW image (only returned if removeBW is False)
    2 - the original image is smaller than IMG_SZ in at least 1 dimension
    """
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    # is the picture dimension at least IMG_SZ for both dimensions?
    if img.shape[0]<IMG_SZ or img.shape[1]<IMG_SZ:
        if verbose:
            print(imgPath+' rejected: too small')
        return [2]

    h,w,c = img.shape
    h = float(h)
    w = float(w)
    num_slides,hskip,wskip,min_dim = 1,0,0,h
    if h>w:
        num_slides = np.ceil(h/w)
        hskip = w - (w*num_slides-h)/(num_slides-1)
        min_dim = w
    elif h<w:
        num_slides = np.ceil(w/h)
        wskip = h - (h*num_slides-w)/(num_slides-1)
        min_dim = h

    imgs = []
    for n in range(int(num_slides)):
        imgs.append(img[int(hskip*n):int(hskip*n+min_dim), int(wskip*n):int(wskip*n+min_dim)])

    returnMsg = 0
    rgb_imgs = []
    for i,im in enumerate(imgs):
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        # is the picture colorful enough? Check the mean value of the saturation.
        mean,std = cv2.meanStdDev(im_hsv)
        if mean[1]<BW_THRESHOLD:
            if removeBW:
                if verbose:
                    print('%d of %s rejected: too little colors' %(i,imgPath))
                continue
            else:
                returnMsg = 1

        rgb_imgs.append(im)

    return [returnMsg] + rgb_imgs

def processPic(dataPack):
    # try:
    inDir, filename, output2File, removeBW = dataPack

    imgPath = os.path.join(inDir, filename)
    outPath_reduced = os.path.join(inDir+'_processed_reduced', filename)
    outPath_line = os.path.join(inDir+'_processed_line', filename)
    outPath_binary = os.path.join(inDir+'_processed_binary', filename)

    # print(imgPath)

    # load the image
    imgs = loadImg(imgPath, removeBW)
    returnMsg = imgs[0]
    imgs = imgs[1:]
    if returnMsg==2:
        return

    for i,img in enumerate(imgs):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if returnMsg==0:
            img_dilate = cv2.erode(gray_img, neiborhood8, iterations=2)
            img_dilate = cv2.dilate(img_dilate, neiborhood8, iterations=4)

            img_diff = cv2.absdiff(gray_img, img_dilate)
            
        elif returnMsg==1:
            img_diff = cv2.bitwise_not(gray_img)

        img_diff = cv2.multiply(img_diff, 3)
        img_line = cv2.bitwise_not(img_diff)

        kernel = np.ones((8,8),np.float32)/64
        img_line = cv2.filter2D(img_line,-1,kernel)
        img_line = cv2.dilate(img_line, neiborhood8, iterations=1)

        # img_binary = cv2.adaptiveThreshold(img_line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
        _,img_binary = cv2.threshold(img_line, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        img_line_tmp = np.zeros_like(img_binary)
        img_line_tmp[img_binary>200] = 255
        img_line = img_line_tmp

        # resize the smaller dimension of the image to IMG_SZ
        ratio = IMG_SZ / np.amin(gray_img.shape)
        img_line = cv2.resize(img_line, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        img_binary = cv2.resize(img_binary, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

        print(type(img_line))
        print(np.max(img_line))
        print(np.min(img_line))
        print(img_line.shape)
        # exit(0)

        # visualization (debugging purpose)
        # cv2.imshow('test',grayImg)
        # cv2.imshow('test2',img_dilate)
        # cv2.imshow('test3',img_diff)

        # cv2.imshow('test4',img_line)
        # cv2.imshow('test5',img_binary)
        # cv2.imshow('test6',img_diff_not2)
        # cv2.waitKey()

        # cv2.destroyAllWindows()

        img_reduced = cv2.resize(img, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        
        if output2File:
            cv2.imwrite(outPath_reduced.replace('.jpg', '_%d.jpg'%i), img_reduced)
            cv2.imwrite(outPath_line.replace('.jpg', '_%d.jpg'%i), img_line)
            cv2.imwrite(outPath_binary.replace('.jpg', '_%d.jpg'%i), img_binary)

    # except:
        # print(filename)

    return img_line,img_binary
        

def preprocessData(folderName, prune=False, output2File=True):
    # if prune==True, the function checks to see if the images have already been converted

    makeDir(folderName+'_processed_reduced')
    makeDir(folderName+'_processed_line')
    makeDir(folderName+'_processed_binary')

    if prune:
        indx2remove = set([int(re.search('[0-9]+',filename).group(0)) for filename in os.listdir(folderName+'_processed_reduced')])

        fileIndx = np.asarray([int(re.search('[0-9]+',filename).group(0)) for filename in os.listdir(folderName)])
        fileIndx = np.sort(fileIndx)

        filenames = list(fileIndx)

        rmIndx = sorted([fileIndx.searchsorted(i) for i in indx2remove], reverse=True)
        for indx in rmIndx:
            del filenames[indx]

        print('Done pruning out the processed files...')
        mapList = [(folderName, str(filename)+'.jpg') for filename in filenames]

    else:
        filenames = os.listdir(folderName)
        mapList = [(folderName, filename, True, True) for filename in filenames]

    p = Pool(POOL_WORKER_COUNT)
    # mapList = (mapList[0],)
    p.map(processPic, mapList)

    print('done...')


if __name__ == "__main__":
    # preprocessData('../images/sampleImg')
    GT2illustration('../results/imgs/unet/iter24', '../results/imgs/gt')