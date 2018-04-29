import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import copy
from progress.bar import Bar

def calcCirc(cont):
	return 4*np.pi*cv2.contourArea(cont)/cv2.arcLength(cont, True)**2

#creating an output directory
try:
	os.mkdir('cells')
except:
	pass

#setting manual threshold values
cellThreshValue = 200
nucThreshValue = 80
otsu = True

try:
	dirs = os.listdir('rawCellImages')
except:
	print("Make sure there is folder called 'rawCellImages' containing input images.")
	quit()

imageData = []

#importing image and making grayscale
noCellsProcessed = 0
for imageFile in dirs:
	if imageFile == ".DS_Store":
		continue

	print("Analysing %s..."%imageFile)
	outputDir = 'cells'
	try:
		os.mkdir(outputDir)
	except:
		pass

	# try:
	# 	img = cv2.imread('images/%s'%imageFile)
	# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# except:
	# 	print("%s isn't an image file!"%imageFile)
	# 	continue
	
	img = cv2.imread('rawCellImages/%s'%imageFile)
	print("\tFinished importing image")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print("\tFinished converting to greyscale")

	#thresholding for cells
	ret, cellThresh = cv2.threshold(gray, cellThreshValue, 255, cv2.THRESH_BINARY_INV)

	#generating contours around cells
	img2, cellCon, hierarchy = cv2.findContours(cellThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#iterate through contours and isolate
	noCellCon = len(cellCon)
	print("\tContours found: %s"%noCellCon)
	bar = Bar('\tProcessing', max=noCellCon)
	for i in range(len(cellCon)):
		cellArea = cv2.contourArea(cellCon[i])
		
		#excluding tiny and big blobs
		if cellArea > 3000 and cellArea < 6000 and calcCirc(cellCon[i])>0.7:
			#excluding cells on edges
			x, y, width, height = cv2.boundingRect(cellCon[i])
			if not (x == 0 or x+width >= img.shape[1] or y == 0 or y+height >= img.shape[0]):
				name = str(noCellsProcessed)

				#creating mask for cells
				rawMask = np.zeros((img.shape[0], img.shape[1], 1), dtype = "uint8")+255
				cv2.drawContours(rawMask, cellCon, i, (255), -1)
				revMask = np.bitwise_not(rawMask)
				res1 = cv2.bitwise_and(img, img, mask = rawMask)
				res2 = res1 + revMask

				newDim = 98
				res3 = cv2.resize(res2[y:y+height, x:x+width], (newDim, newDim))

				#isolating cells
				margin = 1
				rawROI = np.zeros((newDim+2*margin, newDim+2*margin, 3))+255
				rawROI[margin:margin+newDim, margin:margin+newDim] = res3[0:newDim, 0:newDim]

				try:
					os.mkdir("%s/cells"%outputDir)
				except:
					pass

				cv2.imwrite("%s/cells/%s.jpg"%(outputDir,name), rawROI)
				noCellsProcessed+=1
		bar.next()
	bar.finish()

print("Done")