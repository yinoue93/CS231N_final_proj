
import shutil

from utils_rgb2line import *
from utils_misc import *


if __name__ == "__main__":
	# scrapeDir = 'scraped'
	# archiveDir_scrape = 'D:\\Backups\\CS231N_data\\scraped'
	# archiveDir_line = 'D:\\Backups\\CS231N_data\\line'
	# archiveDir_binary = 'D:\\Backups\\CS231N_data\\binary'

	scrapeDir = 'small_dataset/small_scraped'
	archiveDir_scrape = 'small_dataset/compressed_scrape'
	archiveDir_line = 'small_dataset/compressed_line'
	archiveDir_binary = 'small_dataset/compressed_binary'

	# webcrawling (utils_crawler.py)

	# clean up the data (?)
	cleanUpDataset(scrapeDir)

	# save crawled imgs to zip files
	zipDirectory(scrapeDir, outputDirName=archiveDir_scrape, zipFileSz=1024)

	# apply preprocessing per zip file
	tmpDirNum = 0
	for i in range(len(os.listdir(archiveDir_scrape))):
		print('\n'+str(i))

		tmpDir = 'tmp'+str(tmpDirNum)
		try:
			shutil.rmtree(tmpDir)
		except:
			print('directory deletion failed...')

		makeDir(tmpDir)
		if os.listdir(tmpDir)!=[]:
			tmpDirNum += 1
			tmpDir = 'tmp'+str(tmpDirNum)
			makeDir(tmpDir)

		makeDir(tmpDir+'/raw')

		# uncompress the data
		print('uncompressing zip file...')
		zipName = os.path.join(archiveDir_scrape, 'compressed_'+str(i))
		unzipper((zipName, tmpDir+'/raw'))

		# apply preprocessing
		print('applying preprocessing...')
		preprocessData(tmpDir+'/raw')
		
		# save the preprocessing result in hdf5
		print('saving "line" in hdf5...')
		zipDirectory(tmpDir+'/raw_processed_line', outputDirName=archiveDir_line, 
				 	 zipFileSz=1024, originalDir=tmpDir+'/raw_processed_reduced',
				 	 overwrite=False)
		print('saving "binary" in hdf5...')
		zipDirectory(tmpDir+'/raw_processed_binary', outputDirName=archiveDir_binary, 
				 	 zipFileSz=1024, originalDir=tmpDir+'/raw_processed_reduced',
				 	 overwrite=False)
