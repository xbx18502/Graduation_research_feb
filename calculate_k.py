# calculate kiwifruit 's each parts' mua, mus, with equal scale enlarge
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
# import tensorflow as tf
import math
# from scipy.optimize import fsolve
# from scipy.optimize import root
import scipy as scipy
from scipy.optimize import minimize
# from scipy import signal
# from scipy import linalg
# import scipy as scp
# import sympy as  syp
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer
# from tensorflow.keras.models import Sequential
import os
import gc
import random

class process_hyperspec(object):
    
    def __init__(self, path, scat):
        self.path = path
        self.scat = scat
        
    def read_data(self):
        ## Read data from txt
        data=pd.read_fwf(self.path, dtype = object, header = None)## Read data as Pandas DataFrame
        data_8 = data.loc[8,:].str[6:] ## Select data of column of wavelengths

        data_nm_dataframe =pd.read_csv(StringIO(data_8[0]))

        with open(self.path,'r',encoding='utf-8') as f:
            content = f.read()

        flag = 0
        for index in range(len(content)):##find the start point  of time and intensity data, pointer is the 'index'
            if (content[index] == '\n') and (flag != 9):
                flag = flag + 1
            elif (content[index] == '\n') and (flag == 9):
                break
            else:
                continue

        data_ns = pd.read_csv(StringIO(content[index:len(content)]), names = data_nm_dataframe.columns[0:len(set(data_nm_dataframe))]) ## Read data of time and gray 
        img_nm_ns = np.array(data_ns)    ## Transform to numpy array                                image as DataFrame; names is a 640x0 array indicates the wavelengths range
        
        self.data_frame = data_ns
        self.data_array = img_nm_ns
        
    def cal_846nm(self):
        time_total, wavelength_total = self.data_array.shape ## Generate an array contains intensity versus time data on 846nm
        data_846nm = np.zeros((time_total,), dtype = float )

        # for i in range(wavelength_total):
        #     data_846nm = self.data_array[:,i] + data_846nm
        data_846nm = np.sum(self.data_array,axis = 1)

        return data_846nm/wavelength_total
    
    def return_dataframe(self):
        
        return self.data_frame
    
    def return_array(self):
        return self.data_array
    
class data_dict:
    dict0 = {}
    for i in range(1,100):
        dict0[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230210\\常温\\Sample'+str(i)+'.txt'
    dict0[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230210\\常温\\irf00.txt'

    dict1 = {}
    for i in range(1,100):
        dict1[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230211\\常温\\Sample'+str(i)+'.txt'
    dict1[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230211\\常温\\irf00.txt'

    dict2 = {}
    for i in range(1,100):
        dict2[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230212\\常温\\Sample'+str(i)+'.txt'
    dict2[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230212\\常温\\irf00.txt'

    dict3 = {}
    for i in range(1,100):
        dict3[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230213\\常温\\Sample'+str(i)+'.txt'
    dict3[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230213\\常温\\irf00.txt'

    dict4 = {}
    for i in range(1,100):
        dict4[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230215\\常温\\Sample'+str(i)+'.txt'
    dict4[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230215\\常温\\irf00.txt'

    dict5 = {}
    for i in range(1,100):
        dict5[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230217\\常温\\Sample'+str(i)+'.txt'
    dict5[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230217\\常温\\irf00.txt'

    dict6 = {}
    for i in range(1,100):
        dict6[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230218\\常温\\Sample'+str(i)+'.txt'
    dict6[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230218\\常温\\irf00.txt'

    dict7 = {}
    for i in range(1,100):
        dict7[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230219\\常温\\Sample'+str(i)+'.txt'
    dict7[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230219\\常温\\irf00.txt'

    dict8 = {}
    for i in range(1,100):
        dict8[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230221\\常温\\Sample'+str(i)+'.txt'
    dict8[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230221\\常温\\irf00.txt'

    dict9 = {}
    for i in range(1,100):
        dict9[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230223\\常温\\Sample'+str(i)+'.txt'
    dict9[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230223\\常温\\irf00.txt'

    dict10 = {}
    for i in range(1,100):
        dict10[i] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230225\\常温\\Sample'+str(i)+'.txt'
    dict10[0] = 'D:\\files2\\Nagoya\\python\\kiwi_ToF_simulation\\feb\\20230225\\常温\\irf00.txt'

    dictn = [dict0,dict1,dict2,dict3,dict4,dict5,dict6,dict7,dict8,dict9,dict10]    

class analyze(object):
    def __init__(self):

        return
    def read_radius(self):
        data1 = pd.read_excel('./kiwi_data.xlsx',sheet_name=1,header=None)
        data2 = np.array(data1)
        return data2[:,1]

    
    def read_everyday(self,dict0,radius):    
        files = os.listdir(dict0[0][0:-10])   # 读入文件夹
        N = len(files)       # 统计文件夹中的文件个数
        scat = [None]*N
        data_480multi9 = np.zeros((N,480), dtype = float)
        time_index = np.zeros((N,480))
        radius0 = np.zeros((N-1,))
        j=0
        for i in range(0,100):
            if os.path.exists(dict0[i]):
                scat[j] = process_hyperspec(dict0[i], i)
                scat[j].read_data()
                data_480multi9[j,:] = scat[j].cal_846nm()
                time_index[j] = scat[j].return_dataframe().index
                if(i!=0):
                    radius0[j-1] = radius[i-1]
                j=j+1
            else:
                pass
        return data_480multi9,time_index,N,radius0

    def generate_h_y(self,data_480multi9,N):
        kernel0 = np.ones((10,))
        data_smooth = np.zeros((N-1,480))
        
        # for i in range(N-1):
        #     data_smooth[i] = np.convolve(kernel0,data_480multi9[i+1],'same')/np.sum(kernel0)

        def smooth(data):   #函数，用来平滑
            return np.convolve(kernel0,data,'same')/np.sum(kernel0)
        
        data_Y = data_480multi9[1:N]
        data_smooth = np.array([smooth(data_Y[i]) for i in range(data_Y.shape[0])])  #使用隐式循环来遍历

        y = np.zeros((data_smooth.shape[0],data_smooth.shape[1]*2-1))
        y[:,0:data_smooth.shape[1]] = data_smooth
        h = np.convolve(kernel0,data_480multi9[0],'same')/np.sum(kernel0)
        return h,y
##################################################
os.chdir(os.path.dirname(os.path.abspath(__file__)))
###################################################
ana = analyze()
H = []
Y = []
TIME = []
R = []
AMOUNT_EXP = len(data_dict.dictn)

def iterate_AMOUNT_EXP(i):  #定义获取每个 H,Y,TIME,R 的方法
    radius = ana.read_radius()
    data_480multi9,time_index,N,radius0 = ana.read_everyday(data_dict.dictn[i],radius)
    h,y = ana.generate_h_y(data_480multi9,N)

    return h,y,time_index,radius0
H_Y_TIME_R = list(iterate_AMOUNT_EXP(i) for i in range(AMOUNT_EXP))
# for i in range(AMOUNT_EXP):
#     radius = ana.read_radius()
#     data_480multi9,time_index,N,radius0 = ana.read_everyday(dictn[i],radius)
#     h,y = ana.generate_h_y(data_480multi9,N)
#     H.append(h)
#     Y.append(y)
#     TIME.append(time_index)
#     R.append(radius0)
H = [H_Y_TIME_R[i][0] for i in range(len(H_Y_TIME_R))]
Y = [H_Y_TIME_R[i][1] for i in range(len(H_Y_TIME_R))]
TIME = [H_Y_TIME_R[i][2] for i in range(len(H_Y_TIME_R))]
R = [H_Y_TIME_R[i][3] for i in range(len(H_Y_TIME_R))]

##########################################
fig = plt.figure(figsize = (8,8))
for i in range(AMOUNT_EXP):
    plt.subplot(3,4,i+1)
    for j in range(Y[i].shape[0]):
        plt.plot(TIME[0][i]*1e-9,Y[i][j,0:480])
# plt.title.set_text(dict0[i][9:-4])
fig.tight_layout(pad=1.1)

plt.show()