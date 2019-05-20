import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
%matplotlib inline

img_B = pd.read_csv(r"C:\Users\Motegi\Documents\Python Scripts\T0516111007281_CNT00144534_B00_NG\area3\area3_B.csv", header = None, engine='python')
img_B = img_B.drop(columns = img_B.shape[1] - 1, axis = 1)

img_G = pd.read_csv(r"C:\Users\Motegi\Documents\Python Scripts\T0516111007281_CNT00144534_B00_NG\area3\area3_G.csv", header = None, engine='python')
img_G = img_G.drop(columns = img_G.shape[1] - 1, axis = 1)

img_R = pd.read_csv(r"C:\Users\Motegi\Documents\Python Scripts\T0516111007281_CNT00144534_B00_NG\area3\area3_R.csv", header = None, engine='python')
img_R = img_R.drop(columns = img_R.shape[1] - 1, axis = 1)

input_img = np.array(0.299 * img_B.T + 0.587 * img_G.T + 0.114 * img_R.T)
