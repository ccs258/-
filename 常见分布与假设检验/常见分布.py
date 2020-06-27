#2.4.1 生成一组符合特定分布的随机数
import numpy

# 生成大小为1000的符合b(10,0.5)二项分布的样本集
s = numpy.random.binomial(n=10,p=0.5,size=1000)

# 生成大小为1000的符合P(1)的泊松分布的样本集
s = numpy.random.poisson(lam=1,size=1000)

# 生成大小为1000的符合U(0,1)均匀分布的样本集，注意在此方法中边界值为左闭右开区间
s = numpy.random.uniform(low=0,high=1,size=1000)

# 生成大小为1000的符合N(0,1)正态分布的样本集，可以用normal函数自定义均值，标准差，也可以直接使用standard_normal函数
s = numpy.random.normal(loc=0,scale=1,size=1000)
s = numpy.random.standard_normal(size=1000)

# 生成大小为1000的符合E(1/2)指数分布的样本集，注意该方法中的参数为指数分布参数λ的倒数
s = numpy.random.exponential(scale=2,size=1000)


# 除了Numpy，Scipy也提供了一组生成特定分布随机数的方法

# 以均匀分布为例，rvs可用来生成一组随机变量的值
from scipy import stats
stats.uniform.rvs(size=10)

#2.4.2 计算统计分布的PMF和PDF
from scipy import stats

# 计算二项分布B(10,0.5)的PMF
x=range(11)
p=stats.binom.pmf(x, n=10, p=0.5)

# 计算泊松分布P(1)的PMF
x=range(11)
p=stats.poisson.pmf(x, mu=1)

# 计算均匀分布U(0,1)的PDF
x = numpy.linspace(0,1,100)
p= stats.uniform.pdf(x,loc=0, scale=1)

# 计算正态分布N(0,1)的PDF
x = numpy.linspace(-3,3,1000)
p= stats.norm.pdf(x,loc=0, scale=1)

# 计算指数分布E(1)的PDF
x = numpy.linspace(0,10,1000)
p= stats.expon.pdf(x,loc=0,scale=1)


#计算统计分布的CDF
# 以正态分布为例，计算正态分布N(0,1)的CDF
x = numpy.linspace(-3,3,1000)
p = stats.norm.cdf(x,loc=0, scale=1)



#统计分布可视化
#二项分布
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

x = range(11)  # 二项分布成功的次数（X轴）
t = stats.binom.rvs(10,0.5,size=10000) # B(10,0.5)随机抽样10000次
p = stats.binom.pmf(x, 10, 0.5) # B(10,0.5)真实概率质量

fig, ax = plt.subplots(1, 1)
sns.distplot(t,bins=10,hist_kws={'density':True}, kde=False,label = 'Distplot from 10000 samples')
sns.scatterplot(x,p,color='purple')
sns.lineplot(x,p,color='purple',label='True mass density')
plt.title('Binomial distribution')
plt.legend(bbox_to_anchor=(1.05, 1))

#泊松分布
#比较λ=2的泊松分布的真实概率质量和10000次随机抽样的结果
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
x=range(11)
t= stats.poisson.rvs(2,size=10000)
p=stats.poisson.pmf(x, 2)

fig, ax = plt.subplots(1, 1)
sns.distplot(t,bins=10,hist_kws={'density':True}, kde=False,label = 'Distplot from 10000 samples')
sns.scatterplot(x,p,color='purple')
sns.lineplot(x,p,color='purple',label='True mass density')
plt.title('Poisson distribution')
plt.legend()

x=range(50)
fig, ax = plt.subplots()
for  lam in [1,2,5,10,20] :
    p=stats.poisson.pmf(x, lam)
    sns.lineplot(x,p,label='lamda= '+ str(lam))
plt.title('Poisson distribution')
plt.legend()

#均匀分布
#比较U(0,1)的均匀分布的真实概率密度和10000次随机抽样的结果
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
x=numpy.linspace(0,1,100)
t= stats.uniform.rvs(0,1,size=10000)
p=stats.uniform.pdf(x, 0, 1)

fig, ax = plt.subplots(1, 1)
sns.distplot(t,bins=10,hist_kws={'density':True}, kde=False,label = 'Distplot from 10000 samples')

sns.lineplot(x,p,color='purple',label='True mass density')
plt.title('Uniforml distribution')
plt.legend(bbox_to_anchor=(1.05, 1))

#正态分布
#比较N(0,1)的正态分布的真实概率密度和10000次随机抽样的结果
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
x=numpy.linspace(-3,3,100)
t= stats.norm.rvs(0,1,size=10000)
p=stats.norm.pdf(x, 0, 1)

fig, ax = plt.subplots(1, 1)
sns.distplot(t,bins=100,hist_kws={'density':True}, kde=False,label = 'Distplot from 10000 samples')


sns.lineplot(x,p,color='purple',label='True mass density')
plt.title('Normal distribution')
plt.legend(bbox_to_anchor=(1.05, 1))

#比较不同均值和标准差组合的正态分布的概率密度函数
x=numpy.linspace(-6,6,100)
p=stats.norm.pdf(x, 0, 1)
fig, ax = plt.subplots()
for  mean, std in [(0,1),(0,2),(3,1)]:
    p=stats.norm.pdf(x, mean, std)
    sns.lineplot(x,p,label='Mean: '+ str(mean) + ', std: '+ str(std))
plt.title('Normal distribution')
plt.legend()


#比较E(1)的指数分布的真实概率密度和10000次随机抽样的结果
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
x=numpy.linspace(0,10,100)
t= stats.expon.rvs(0,1,size=10000)
p=stats.expon.pdf(x, 0, 1)

fig, ax = plt.subplots(1, 1)
sns.distplot(t,bins=100,hist_kws={'density':True}, kde=False,label = 'Distplot from 10000 samples')


sns.lineplot(x,p,color='purple',label='True mass density')
plt.title('Exponential distribution')
plt.legend(bbox_to_anchor=(1, 1))

#比较不同参数的指数分布的概率密度函数
x=numpy.linspace(0,10,100)
fig, ax = plt.subplots()
for  scale in [0.2,0.5,1,2,5] :
    p=stats.expon.pdf(x, scale=scale)
    sns.lineplot(x,p,label='lamda= '+ str(1/scale))
plt.title('Exponential distribution')
plt.legend()

#假设检验


#正态检验 #检验分布是否是正态分布

import numpy as np
from scipy.stats import shapiro
data_nonnormal = np.random.exponential(size=100)
data_normal = np.random.normal(size=100)

def normal_judge(data):
    stat, p = shapiro(data)
    if p > 0.05:
        return 'stat={:.3f}, p = {:.3f}, probably gaussian'.format(stat,p)
    else:
        return 'stat={:.3f}, p = {:.3f}, probably not gaussian'.format(stat,p)

# output
normal_judge(data_nonnormal)
# 'stat=0.850, p = 0.000, probably not gaussian'
normal_judge(data_normal)
# 'stat=0.987, p = 0.415, probably gaussian'


#卡方检验  #目的：检验两组类别变量是相关的还是独立的

from scipy.stats import chi2_contingency
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')

# output
#stat=0.272, p=0.873
#Probably independent


#T检验--检验两个独立样本集的均值是否具有显著差异
from scipy.stats import ttest_ind
import numpy as np
data1 = np.random.normal(size=10)
data2 = np.random.normal(size=10)
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# output
# stat=-1.382, p=0.184
# Probably the same distribution

#ANOVA
#与t-test类似，ANOVA可以检验两组及以上独立样本集的均值是否具有显著差异
from scipy.stats import f_oneway
import numpy as np
data1 = np.random.normal(size=10)
data2 = np.random.normal(size=10)
data3 = np.random.normal(size=10)
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# output
# stat=0.189, p=0.829
# Probably the same distribution


#Mann-Whitney U Test ~检验两个样本集的分布是否相同
from scipy.stats import mannwhitneyu
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# output
# stat=40.000, p=0.236
# Probably the same distribution

