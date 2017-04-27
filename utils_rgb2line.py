import numpy as np
import cv2
import glob
import os

IMG_SZ = 256.0
BW_THRESHOLD = 15

# much of the code borrowed from http://qiita.com/khsk/items/6cf4bae0166e4b12b942#_reference-d2b636028b7b8fbf4b34

neiborhood8 = np.array([[1, 1, 1],
						[1, 1, 1],
						[1, 1, 1]],
						np.uint8)

def loadImg(imgPath):
	img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# is the picture dimension at least IMG_SZ for both dimensions?
	if img.shape[0]<IMG_SZ or img.shape[1]<IMG_SZ:
		print imgPath+' rejected: too small'
		return None

	# is the picture colorful enough? Check the mean value of the saturation.
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	mean,std = cv2.meanStdDev(img_hsv)
	print imgPath
	print mean[0]
	if mean[0]<BW_THRESHOLD:
		print imgPath+' rejected: too little colors'
		return None
	
	return img_gray

def processPic(imgPath):
	# load the image
	img = loadImg(imgPath)
	if img==None:
		return

	# resize the smaller dimension of the image to IMG_SZ
	ratio = IMG_SZ / np.amin(img.shape)

	img_dilate = cv2.erode(img, neiborhood8, iterations=2)
	img_dilate = cv2.dilate(img_dilate, neiborhood8, iterations=4)

	img_diff = cv2.absdiff(img, img_dilate)

	img_diff = cv2.multiply(img_diff, 3)

	img_diff_not = cv2.bitwise_not(img_diff)
	img_diff_not = cv2.resize(img_diff_not, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

	at = cv2.adaptiveThreshold(img_diff_not, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 8)

	# gray = c.cvtColor(img_diff_not, c.COLOR_RGB2GRAY)


	# cv2.imwrite(os.path.dirname(path) + '_clean_senga_color_gray/' + os.path.basename(path), img_diff_not)

	# cv2.imshow('test',img)
	# cv2.imshow('test2',img_dilate)
	# cv2.imshow('test3',img_diff)

	cv2.imshow('test4',img_diff_not)
	cv2.imshow('test5',at)
	# cv2.imshow('test6',img_diff_not2)
	cv2.waitKey()

	cv2.destroyAllWindows()


if __name__ == "__main__":
	folderName = 'test_imgs'

	for path in os.listdir(folderName):
		absPath = os.path.join(folderName, path)
		# absPath = os.path.join(folderName, '2071098.jpg')


		processPic(absPath)
		# exit()