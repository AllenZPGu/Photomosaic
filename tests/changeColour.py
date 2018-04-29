import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import copy
from progress.bar import Bar

img = cv2.imread('0.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cellThreshValue = 200
ret, cellThresh = cv2.threshold(gray, cellThreshValue, 255, cv2.THRESH_BINARY_INV)

img2, cellCon, hierarchy = cv2.findContours(cellThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#iterate through contours and isolate
noCellCon = len(cellCon)
print("\tContours found: %s"%noCellCon)
bar = Bar('Processing', max=noCellCon)
for i in range(len(cellCon)):
	if i>0:
		continue
	x, y, width, height = cv2.boundingRect(cellCon[i])

	#creating mask for cells
	rawMask = np.zeros((img.shape[0], img.shape[1], 1), dtype = "uint8")
	cv2.drawContours(rawMask, cellCon, i, (255), -1)
	revMask = np.bitwise_not(rawMask)
	res1 = cv2.bitwise_and(img, img, mask = rawMask)
	res2 = res1 + revMask

	#isolating cells
	margin = 5
	rawROI = np.zeros((height+2*margin, width+2*margin, 3))
	rawROI[margin:margin+height, margin:margin+width] = res2[y:y+height, x:x+width]

	cv2.imwrite("sad%s.jpg"%i, rawROI)

	grayROI = cv2.cvtColor(rawROI.astype(np.uint8), cv2.COLOR_BGR2GRAY)
	grayAvg = np.average(grayROI[grayROI<250])
	print(grayAvg)
	newAvg = np.array([0,0,255])
	rawROI3 = newAvg/grayAvg * cv2.cvtColor(grayROI, cv2.COLOR_GRAY2BGR)

	cv2.imwrite("sadddddddd%s.jpg"%i, rawROI3)