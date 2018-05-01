import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import copy
from progress.bar import Bar

img = cv2.imread('0.jpg')

resized_img = cv2.resize(img, dsize=(500,50))

cv2.imwrite('lmao.jpg', resized_img)