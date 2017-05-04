import random
import os
import cv2
import sys
import numpy as np
import re
import h5py

import constants

from shutil import copyfile
from utils_rgb2line import loadImg,makeDir
from scipy.misc import imread,imsave

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
			print imgPath
			copyfile(imgPath, imgPath.replace(folderName,outfolder))

			if counter==num_imgs:
				break

	print counter/float(i)

def findMean(dataDir):
	m = 0
	fnames = os.listdir(dataDir+'raw_processed_reduced'+'\\')
	fnameLen = float(len(fnames))

	dirMod = ['binary','line','reduced']
	for mod in dirMod:
		avgR,avgG,avgB = 0,0,0
		for i,fname in enumerate(fnames):
			if i%int(fnameLen*0.1)==1:
				print '. R: %f, G: %f, B: %f' %(avgR*fnameLen/i, avgG*fnameLen/i, avgB*fnameLen/i)
			imgs = imread(dataDir+'raw_processed_'+mod+'\\'+fname)
			if len(imgs.shape)==3:
				avgR += np.mean(imgs[:,:,0])/fnameLen
				avgG += np.mean(imgs[:,:,1])/fnameLen
				avgB += np.mean(imgs[:,:,2])/fnameLen
			else:
				avgR += np.mean(imgs[:,:])/fnameLen

		print mod
		print 'R: %f' % avgR
		print 'G: %f' % avgG
		print 'B: %f' % avgB


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

def numpy2jpg(outputFname, arr, meanVal=0):
	if meanVal==0:
		outputImg = arr
	elif len(arr.shape)==3:
		outputImg = arr + [REDUCED_R_MEAN,REDUCED_G_MEAN,REDUCED_B_MEAN]
	else:
		outputImg = arr + meanVal

	imsave(outputFname, outputImg)

def h52numpy(hdf5Filename, outputDir='', checkMean=False, batch_sz=1):
	"""
	Returns a shuffled list of input and output data, with each element of the list
	numpy array of shape (batch_sz, H, W, C)
	"""
	if outputDir:
		makeDir(outputDir)

	if 'line' in hdf5Filename:
		input_mean = LINE_MEAN
	elif 'binary' in hdf5Filename:
		input_mean = BINARY_MEAN

	meanTotal = np.asarray([0]*4)
	count = 0
	inData = []
	outData = []
	with h5py.File(hdf5Filename,'r') as hf:
		keys = hf.keys()
		random.shuffle(keys)

		tmpIn = np.empty(shape=(batch_sz, IMG_DIM, IMG_DIM, 1), dtype=int)
		tmpOut = np.empty(shape=(batch_sz, IMG_DIM, IMG_DIM, 3), dtype=int)
		for i,key in enumerate(keys):
			indx = i%batch_sz
			data = hf.get(key)
			tmpIn[indx,:,:,:] = data[:,:,0].astype(int) - input_mean
			tmpOut[indx,:,:,:] = data[:,:,1:].astype(int) - [REDUCED_R_MEAN,REDUCED_G_MEAN,REDUCED_B_MEAN]

			if indx==batch_sz-1:
				inData.append(tmpIn)
				outData.append(tmpOut)

			if outputDir:
				numpy2jpg(os.path.join(outputDir, key+'_in.png'), data[:,:,0])
				numpy2jpg(os.path.join(outputDir, key+'_out.png'), data[:,:,1:])

			if checkMean:
				meanTotal[0] += np.mean(inData[count])
				meanTotal[1] += np.mean(outData[count][:,:,0])
				meanTotal[2] += np.mean(outData[count][:,:,1])
				meanTotal[3] += np.mean(outData[count][:,:,2])

				count += 1

	if checkMean:
		print count
		print meanTotal
		print 'Means: ' + str(meanTotal/count)

	return inData,outData

#--------------------------END H5 MODULES---------------------------------


import Image
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
		print imgPath
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
import shutil
from multiprocessing import Pool
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

	print "Zipping up %s to %s" %(dataDir,outputDirName)

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
				print '.'

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

	print 'Processes dispatched...'

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
	print 'Unzipping the directory %s' % dataDir
	unzipDirectory(dataDir)
	print 'Unzipping successful....'

#--------------------------END ZIPPING/UNZIPPING MODULE---------------------------------

if __name__ == "__main__":
	# getBWPics()
	# zipDirectory('test_scraped_processed_binary', outputDirName='D:\\Backups\\CS231N_data\\processed_binary', zipFileSz=1024)
	# unzipDirectory('test_scraped_compressed', 'test_scraped_uncompressed')

	# cleanUpDataset('test_scraped')

	# zipDirectory('test_imgs', outputDirName='D:\\Backups\\CS231N_data\\processed_lines_np', 
	# 			 zipFileSz=1024, pic2h5=True)
	

	# zipDirectory('small_scraped_processed_line', outputDirName='tmp', 
	# 			 zipFileSz=1024, originalDir='small_scraped_processed_reduced')
	# h52numpy('D:\\Backups\\CS231N_data\\line\\line_dataset_26', 'tmp3')

	# unzipDirectory('D:\\Backups\\CS231N_data\\scraped', outputDir='scraped')

	# findMean('D:\\Backups\\CS231N_data\\tmp\\')
	
	# unzipper(('D:\\Backups\\CS231N_data\\scraped\\compressed_26', 'tmp4'))

	pass