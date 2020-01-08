'''
获取 ＋ 预处理 数据
'''
import os
import sys
import numpy as np
import csv
import math


def just_a_test() :
    print ('push to new master')

def getdata (path, num_of_row) :
    '''
    param path : str, 数据的路径 .csv
    param num_of_row :  int, 前多少行是输入的特征 (多少行是用来预测的特征)
    return X, Y : numpy.array, 返回特征 ＋ 目标结果
    '''
    # a = np.loadtxt(path)
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    # print (rows)#输出所有数据
    a=np.array(rows)#rows是数据类型是‘list',转化为数组类型好处理

    l = (a.shape)[0]
    X = np.zeros ([l, num_of_row])
    Y = np.zeros ([l, 1])
    for i in range (l) :
        X[i][:] = a[i][:num_of_row]
        Y[i][:] = a[i][num_of_row:]

    return X,Y

def normalization (x, num_of_row) :
    '''
    对数据进行预处理， 只是进行简单的白化 (标准化的一种 减均值 除方差) 操作 x = (x - mean) / std
    params x: numpy array, 希望被标准化的数组
    params num_of_row : int , 需要进行标准化的列数 （因为是对每一个特征单独做标准化）
    return x: numpy array, 标准化后的数组
    '''
    for i in range (num_of_row) :
        mean = np.average(x[:][i])
        std = np.std (x[:][i])
        x[:][i] = (x[:][i] - mean) / std
    return x

def split_train_test_data (X,Y) :
    '''
    对数据集进行划分 按照 4 : 1 进行划分
    80 % 的数据进行训练 20 % 用于测试 比例可以自行调整
    params X : numpy array, 全部数据
    params Y : numpy array, 数据的标记
    return x_train,y_train, x_test, y_test : numpy array 返回训练集和测试集
    '''
    l = len (X)
    part_mid_point = (int) (l * 4 / 5)
    x_train = X[:part_mid_point]
    y_train = Y[:part_mid_point]
    x_test = X[part_mid_point:]
    y_test = Y[part_mid_point:]
    return x_train, y_train, x_test, y_test



if __name__ == '__main__' :
    '''
    if __name__ == '__main__' 这种方法
    经常用于测试程序 ！
    '''
    path = "000036.csv"
    # X, Y = getdata (path, 13)
    X,Y = getdata (path, 8)
    # print (Y)
    normalization (X, 8)
    # print (Y)
    split_train_test_data (X,Y)
