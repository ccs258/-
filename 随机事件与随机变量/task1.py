# -*- coding: utf-8 -*-
# @Time : 2020/6/21 11:12
# @Author : ccs

import pylab as pl
import numpy as np
from scipy import stats
import random
import math
import pandas as pd
#二项分布
class Task:
    def binom(self):

        n =  10
        k = np.arange(n+1)
        pcoin = stats.binom.pmf(k,n,0.5)
        pl.stem(k,pcoin,basefmt='k-')
        pl.margins(0,1)
        pl.savefig('./task1.png')

        pl.show()

    def convx(self):
        # 协方差
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
class CalBayes:
    def separate_by_class(self,dataset):
        """

        step1.按类别划分样本
        separated = {0: [[att1, att2, ... att8, 0], ...],
                    1: [[att1, att2, ... att8, 1], [att1, att2, ... att8, 1], ...]}

        """
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = [] #构造类别字典，key为类别，value为该类别下的所有可能的属性取值矩阵
            separated[vector[-1]].append(vector)
        return separated

    def mean(self,numbers):
        return sum(numbers)/float(len(numbers))

    def stedv(self,numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

    def summarize(self,dataset):
        """
        step2:计算每一个属性(按列)的均值方差
        注意这个地方zip(*dataset)的用法
        """

        summaries = [(self.mean(attribute),self.stedv(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries
    def summarize_by_class(self,dataset):
        """
        step3：按类别提取属性特征
        #summaries = {0:[(att1_mean,att1_stdev), (att2_mean,att2_stdev), .., (att8_mean,att8_stdev)],
             1:[(att1_mean,att1_stdev), (att2_mean,att2_stdev), .., (att8_mean,att8_stdev)]}

        """
        seperated = self.separate_by_class(dataset)
        summaries = {}
        keyList = list(seperated.keys())
        for classValue in keyList:
            summaries[classValue] = self.summarize(seperated[classValue])
        return summaries

    def calculate_probability(self,x,mean,stdev):
        """
        step4：计算高斯概率密度函数. 计算样本的某一属性x的概率,归属于某个类的似然

        注意理解高斯概率密度函数：对应是函数；而高斯概率密度函数基于X轴求面积，则为概率累计分布
        """
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1/(math.sqrt(2*math.pi)*stdev))*exponent


    def calculate_class_probalities(self,summaries,inputVector):
        """
        step5:对一个样本,计算它属于每个类的概率
        """
        probabilities = {}
        keyList = list(summaries.keys())
        for classValue in keyList:
            probabilities[classValue] = 1
            for i in range(len(summaries[classValue])):
                mean,stdev = summaries[classValue][i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculate_probability(x,mean,stdev)
        return probabilities

    def predict(self,summaries,inputVector):
        """
        step6:单个数据样本的预测. 找到最大的概率值,返回关联的类
        """
        probalities = self.calculate_class_probalities(summaries,inputVector)
        bestLabel,bestProb = None,-1
        keyList = list(probalities.keys())
        for classValue in keyList:
            if bestLabel is None or probalities[classValue] > bestProb:
                bestProb = probalities[classValue]
                bestLabel = classValue
        return bestLabel

    def get_predictions(self,summaries,testSet):
        """
        step7:多个数据样本的预测
        """
        predictions = []
        for i in range(len(testSet)):
            result = self.predict(summaries,testSet[i])
            predictions.append(result)
        return predictions

    def get_accuracy(self,testSet,preditions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == preditions[x]:
                correct += 1
        return (correct/float(len(testSet)))*100.0

    def main(self):
        # 读取数据
        filename = 'pima-indians-diabetes.data.csv'
        dataset = pd.read_csv(filename, header=None)
        dataset = np.array(dataset)

        # 随机划分数据:67%训练和33%测试
        trainSize = int(len(dataset) * 2 / 3)  # (512,9)(256,9)
        randomIdx = [i for i in range(len(dataset))]
        random.shuffle(randomIdx)
        trainSet = []
        testSet = []
        trainSet.extend(dataset[idx, :] for idx in randomIdx[:trainSize])
        testSet.extend(dataset[idx, :] for idx in randomIdx[trainSize:])

        # 计算模型
        summaries = self.summarize_by_class(trainSet)

        # 用测试数据集测试模型
        predictions = self.get_predictions(summaries, testSet)
        accuracy = self.get_accuracy(testSet, predictions)
        print(('Accuracy:{0}%').format(accuracy))

if __name__ == '__main__':
    cal_bayes = CalBayes()
    cal_bayes.main()

