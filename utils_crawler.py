import sys, os, shutil

from argparse import ArgumentParser
import re

import urllib
import urllib2
from bs4 import BeautifulSoup
from urlparse import urljoin
from posixpath import basename

import threading

max_thread = 10
sema = threading.Semaphore(max_thread)
lock = threading.Lock()

def downloadPic(url, folderName):
	try:
		response = urllib2.urlopen(url, timeout=10)

		soup = BeautifulSoup(response.read(), "html.parser")

		picObj = soup.find('div', id='large')
		linkObj = picObj.find('img')
		imgUrl = urljoin(url, linkObj['src'])

		outputName = '%s\\%s.jpg' %(folderName,basename(url))
		urllib.urlretrieve(imgUrl, outputName)

	except:
		# print url
		pass

	sema.release()

def scrape_page(url, folderName):
	for i in range(10):
		try:
			response = urllib2.urlopen(url, timeout=10)
		except:
			print '-'*20+'FAILED'+'-'*20

	soup = BeautifulSoup(response.read(), "html.parser")

	picList = soup.find('ul', id='thumbs2')
	for pic in picList.findAll('li'):
		linkObj = pic.find('a', href=True)
		linkUrl = urljoin(url, linkObj['href'])

		try:
			downloadPic(linkUrl, folderName)
		except:
			print '-'*20+'FAILED'+'-'*20
			return

	sema.release()

def scrape(base_url, folderName, scrapePattern):
	thread_list = []
	# scrape method #1 - page -> image page -> image
	# for curr_page in xrange(1000, 5000):
	# 	print curr_page

	# 	sema.acquire(True)
	# 	url = '%s?p=%d' %(base_url,curr_page)

	# 	th = threading.Thread(target=scrape_page, 
	# 						  args=(url,folderName))

	# 	thread_list.append(th)
	# 	th.start()

	# # wait for the threads to finish
	# for th in thread_list:
	# 	th.join()

	# scrape method #2 - image page -> image
	# for curr_img in reversed(xrange(1000000, 1248000)):
	for curr_img in reversed(xrange(1000000, 1248510)):
		if os.path.isfile(os.path.join(folderName, '%d.jpg'%curr_img)):
			continue
		sema.acquire(True)
		url = '%s%d' %(base_url,curr_img)

		th = threading.Thread(target=downloadPic, 
							  args=(url,folderName))

		thread_list.append(th)
		th.start()

	# wait for the threads to finish
	for th in thread_list:
		th.join()


def parseCLI():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)

	parser = ArgumentParser(description=desc)

	parser.add_argument('-u', '--url', type = str, dest = 'url', required = True,
						help = 'URL of the website to crawl in')
	parser.add_argument('-f', '--folderName', type = str, dest = 'folderName', required = True,
						help = 'Crawl output folder')

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parseCLI()

	# make the directory for the created files
	try:
		os.makedirs(args.folderName)
	except:
		pass

	scrapePattern = ''
	scrape(args.url, args.folderName, scrapePattern)
	
	print "done"