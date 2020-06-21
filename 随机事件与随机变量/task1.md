#理论部分~基本全部理解

    掌握以下理论部分，以及公式实现；
    基本概念：随机事件，样本空间等；
    概率基础：古典概型，条件概率，贝叶斯公式；
    随机变量及其分布特征
    另外，还另外补充了均值，方差等基础数值运算方法的学习。



#练习部分


##python实现二项分布
        import pylab as pl
        import numpy as np
        from scipy import stats
        n =  10
        k = np.arange(n+1)
        pcoin = stats.binom.pmf(k,n,0.5)
        pl.stem(k,pcoin,basefmt='k-')
        pl.margins(0,1)
        pl.show()
        pl.savefig('./task1.png')
        
        过程理解及图像结果说明：
        上述图像结果横坐标表示二项分布试验次数；纵坐标表示二项分布的值；可以看出，试验次数为10时，二项分布中每次事件发生概率为0。5，独立重复10次试验中，
        可能出现的情况是：（0,10）即表示0次正面，1次反面；(2,8)即表示2次正面，8次反面；...；(5,5)表示5次正面，5次反面；...;(10,0)表示10次正面，0次反面；
        再根据二项分布概率分布计算公式，得出上述每一种情况的结果，即表示对应的二项分布值，所有值描点连接即可得到曲线。

        pylab的画图及保存图片的参考代码：
        import pylab
        pylab.ion()
        x = pylab.arange( 0, 10, 0.1)
        y = pylab.sin(x)
        pylab.plot(x,y, 'ro-')
        pylab.show() 
        pylab.savefig('temp.png')

        pylab.savefig保存图片为空白图片，原因是在show后面是新生成的文件句柄，挪到前面去即可：
        其实产生这个现象的原因很简单：在 plt.show() 后调用了 plt.savefig() ，在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候你再 plt.savefig() 就会保存这个新生成的空白图片。
        https://blog.csdn.net/u010099080/article/details/52912439




##协方差和相关系数
    import numpy as np
    
    # 随机生成两个样本
    x = np.random.randint(0, 9, 1000)
    y = np.random.randint(0, 9, 1000)
    
    # 计算平均值
    mx = x.mean()
    my = y.mean()
    
    # 计算标准差
    stdx = x.std()
    stdy = y.std()
    
    # 计算协方差矩阵
    covxy = np.cov(x, y)
    print(covxy)
    
    # 我们可以手动进行验证
    # covx等于covxy[0, 0], covy等于covxy[1, 1]
    # 我们这里的计算结果应该是约等于，因为我们在计算的时候是使用的总体方差(总体方差和样本方差是稍微有点区别的)
    covx = np.mean((x - x.mean()) ** 2) 
    covy = np.mean((y - y.mean()) ** 2) 
    print(covx)
    print(covy)
    # 这里计算的covxy等于上面的covxy[0, 1]和covxy[1, 0]，三者相等
    covxy = np.mean((x - x.mean()) * (y - y.mean()))
    print(covxy)
    
    # 下面计算的是相关系数矩阵(和上面的协方差矩阵是类似的)
    coefxy = np.corrcoef(x, y)
    print(coefxy)

##贝叶斯公式；
    见task1.py
## 贝叶斯用到的数据简介
    数据集简介
    该数据集最初来自国家糖尿病/消化/肾脏疾病研究所。数据集的目标是基于数据集中包含的某些诊断测量来诊断性的预测 患者是否患有糖尿病。
    从较大的数据库中选择这些实例有几个约束条件。尤其是，这里的所有患者都是Pima印第安至少21岁的女性。
    数据集由多个医学预测变量和一个目标变量组成Outcome。预测变量包括患者的怀孕次数、BMI、胰岛素水平、年龄等。
    
    【1】Pregnancies：怀孕次数
    【2】Glucose：葡萄糖
    【3】BloodPressure：血压 (mm Hg)
    【4】SkinThickness：皮层厚度 (mm)
    【5】Insulin：胰岛素 2小时血清胰岛素（mu U / ml
    【6】BMI：体重指数 （体重/身高）^2
    【7】DiabetesPedigreeFunction：糖尿病谱系功能
    【8】Age：年龄 （岁）
    【9】Outcome：类标变量 （0或1）

    版权声明：本文为CSDN博主「易悠」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/yizheyouye/article/details/79791473

#参考
    使用Python计算方差协方差相关系数：https://blog.csdn.net/theonegis/article/details/85059105
    【Python】解决使用 plt.savefig 保存图片时一片空白：https://blog.csdn.net/u010099080/article/details/52912439
    pylab.show()没有显示图形图像（python的matplotlib画图包）：https://www.cnblogs.com/emanlee/p/4384233.html
    
    贝叶斯分类器(Python实现+详细完整源码和原理)：https://blog.csdn.net/qq_25948717/article/details/81744277
    使用python进行贝叶斯统计分析：https://goldengrape.github.io/posts/python/shi-yong-pythonjin-xing-bei-xie-si-tong-ji-fen-xi/

    终极参考：
    用Python学习朴素贝叶斯分类器：https://blog.csdn.net/index20001/article/details/73925325