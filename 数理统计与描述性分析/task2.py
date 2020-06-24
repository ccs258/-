# -*- coding: utf-8 -*-
# @Time : 2020/6/24 16:33
# @Author : ccs
#NumPy系统是Python的一种开源的数值计算扩展。用来存储和处理大型矩阵。

#数值统计
import numpy as np
a = [1,2,4,5,3,12,12,23,43,52,11,22,22,22]
a_mean = np.mean(a)  #均值
a_med = np.median(a)  #中位数
print("a的平均数:",a_mean)
print("a的中位数:",a_med)
#------------------------------------------------------------
from scipy import stats
'''
Scipy是一个高级的科学计算库，Scipy一般都是操控Numpy数组来进行科学计算，
Scipy包含的功能有最优化、线性代数、积分、插值、拟合、特殊函数、快速傅里叶变换、
信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。
'''
a_m1 =stats.mode(a)[0][0]
print("a的众数1:",a_m1)
#-------------------------------------------------------------
import pandas as pd
#将一维数组转成Pandas的Series，然后调用Pandas的mode()方法
ser = pd.Series(a)
a_m2 = ser.mode()
print("a的众数2:",a_m2)


#数值统计
import numpy as np
a = [1,2,4,5,3,12,12,23,43,52,11,22,22,22]
a_var = np.var(a)  #方差
a_std1 = np.sqrt(a_var) #标准差
a_std2 = np.std(a) #标准差
a_mean = np.mean(a)  #均值
a_cv =  a_std2 /a_mean #变异系数
print("a的方差:",a_var)
print("a的方差:",a_std1)
print("a的方差:",a_std2)
print("a的变异系数:",a_cv)


#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = list(np.random.randn(10000))
#生成标准正态分布的随机数（10000个）
plt.hist(data,1000,facecolor='g',alpha=0.5)
'''
plt.hist(arr, bins=10, facecolor, edgecolor,alpha，histtype='bar')
bins：直方图的柱数，可选项，默认为10
alpha: 透明度
'''
plt.show()
s = pd.Series(data) #将数组转化为序列
print('偏度系数',s.skew())
print('峰度系数',s.kurt())


