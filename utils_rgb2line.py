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

neiborhood8 = np.array([[1, 1, 1],
						[1, 1, 1],
						[1, 1, 1]],
						np.uint8)

def loadImg(imgPath, verbose=False):
	img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

	# is the picture dimension at least IMG_SZ for both dimensions?
	if img.shape[0]<IMG_SZ or img.shape[1]<IMG_SZ:
		if verbose:
			print imgPath+' rejected: too small'
		return []

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
		imgs.append(img[hskip*n:(hskip*n+min_dim), wskip*n:(wskip*n+min_dim)])

	rgb_imgs = []
	for i,im in enumerate(imgs):
		im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

		# is the picture colorful enough? Check the mean value of the saturation.
		mean,std = cv2.meanStdDev(im_hsv)
		if mean[1]<BW_THRESHOLD:
			if verbose:
				print '%d of %s rejected: too little colors' %(i,imgPath)
			continue

		rgb_imgs.append(im)
	
	return rgb_imgs

def makeDir(dirname):
	# make the directory for the created files
	try:
		os.makedirs(dirname)
	except:
		pass

def processPic(dataPack):
	try:
		inDir, filename = dataPack

		imgPath = os.path.join(inDir, filename)
		outPath_reduced = os.path.join(inDir+'_processed_reduced', filename)
		outPath_line = os.path.join(inDir+'_processed_line', filename)
		outPath_binary = os.path.join(inDir+'_processed_binary', filename)

		# print imgPath

		# load the image
		imgs = loadImg(imgPath)
		if not len(imgs):
			return

		for i,img in enumerate(imgs):
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			img_dilate = cv2.erode(gray_img, neiborhood8, iterations=2)
			img_dilate = cv2.dilate(img_dilate, neiborhood8, iterations=4)

			img_diff = cv2.absdiff(gray_img, img_dilate)
			
			img_diff = cv2.multiply(img_diff, 3)
			img_line = cv2.bitwise_not(img_diff)

			# img_binary = cv2.adaptiveThreshold(img_line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
			_,img_binary = cv2.threshold(img_line, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			# resize the smaller dimension of the image to IMG_SZ
			ratio = IMG_SZ / np.amin(gray_img.shape)
			img_line = cv2.resize(img_line, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
			img_binary = cv2.resize(img_binary, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

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
			cv2.imwrite(outPath_reduced.replace('.jpg', '_%d.jpg'%i), img_reduced)
			cv2.imwrite(outPath_line.replace('.jpg', '_%d.jpg'%i), img_line)
			cv2.imwrite(outPath_binary.replace('.jpg', '_%d.jpg'%i), img_binary)

			i += 1
	except:
		print filename
		

def preprocessData(folderName):
	makeDir(folderName+'_processed_reduced')
	makeDir(folderName+'_processed_line')
	makeDir(folderName+'_processed_binary')

	indx2remove = set([int(re.search('[0-9]+',filename).group(0)) for filename in os.listdir(folderName+'_processed_reduced')])

	fileIndx = np.asarray([int(re.search('[0-9]+',filename).group(0)) for filename in os.listdir(folderName)])
	fileIndx = np.sort(fileIndx)

	rmIndx = sorted([fileIndx.searchsorted(i) for i in indx2remove], reverse=True)

	filenames = list(fileIndx)

	for indx in rmIndx:
		del filenames[indx]

	print 'Done pruning out the processed files...'

	p = Pool(POOL_WORKER_COUNT)
	mapList = [(folderName, str(filename)+'.jpg') for filename in filenames]
	# mapList = (mapList[0],)
	p.map(processPic, mapList)

	print 'done...'

if __name__ == "__main__":
	preprocessData('small_scraped')