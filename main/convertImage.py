import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import copy
from progress.bar import Bar
import random

def sucPyrDown(img, n):
	for i in range(n):
		aimg = cv2.pyrDown(img)
		img = copy.copy(aimg)
	return img

inputDir = 'inputImages'
cellsDir = 'cells/cells'
outputDir = 'outputImages'

try:
	inputDirs = os.listdir(inputDir)
	cellsDirs = os.listdir(cellsDir)
except:
	print("Make sure there are folders called %s and %s."%(inputDir, cellsDir))
	quit()

try:
	os.mkdir(outputDir)
except:
	pass

totDim = 100
cellsH = 80
for imageFile in inputDirs:
	if imageFile == ".DS_Store":
		continue
	print("Converting %s ..."%imageFile)
	img = cv2.imread('%s/%s'%(inputDir, imageFile))
	dimg = cv2.resize(img, (int(img.shape[1]/img.shape[0]*cellsH), cellsH))
	dimgW = dimg.shape[1]
	dimgH = dimg.shape[0]
	rawCanvas = np.zeros((dimgH*totDim, dimgW*totDim, 3))
	
	bar = Bar('\tProcessing', max=np.shape(dimg)[0]*np.shape(dimg)[1])
	for row in range(np.shape(dimg)[0]):
		for pixel in range(np.shape(dimg)[1]):
			dimgAvg = dimg[row, pixel]

			cellPic = random.choice(cellsDirs)
			cell = cv2.imread('%s/%s'%(cellsDir, cellPic))
			grayCell = cv2.cvtColor(cell.astype(np.uint8), cv2.COLOR_BGR2GRAY)
			grayAvg = np.average(grayCell[grayCell<250])

			
			colCell = dimgAvg/grayAvg * cv2.cvtColor(grayCell, cv2.COLOR_GRAY2BGR)

			#creating a maxk
			ret, cellThresh = cv2.threshold(grayCell, 250, 255, cv2.THRESH_BINARY_INV)
			img2, cellConRaw, hierarchy = cv2.findContours(cellThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cellCon = sorted(cellConRaw, key=lambda x: -cv2.contourArea(x))
			
			rawMask = np.zeros((totDim, totDim, 1), dtype = "uint8")
			cv2.drawContours(rawMask, cellCon, 0, (255), -1)
			revMask = np.bitwise_not(rawMask)
			res1 = cv2.bitwise_and(colCell, colCell, mask = rawMask)
			res2 = res1 + revMask

			rawCanvas[row*totDim:(row+1)*totDim, pixel*totDim:(pixel+1)*totDim] = res2
			bar.next()
	bar.finish()

	cv2.imwrite("%s/%s.jpg"%(outputDir,'%s_mosaic'%imageFile[:-4]), rawCanvas)
print("Done")