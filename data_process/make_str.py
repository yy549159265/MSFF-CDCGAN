import os
from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from queue import Queue, LifoQueue, PriorityQueue



# 返回 RNA 二级结构一维数组
def helper4(path,lab_path):
    result = []
    files = []
    B =[]
    with open('..//lable//'+lab_path+'//'+lab_path+'.txt') as f:
        for line in f.readlines():
            a = line.strip('\n')
            a = a.split("*")
            b = a[0].rstrip('.ct')
            B.append(b)
            fname = path + a[0]
            files.append(fname)
        for file in files:
            fr = open(file)
            length = fr.readline().split('\t')[0]  # 获取序列长度
            dict = {}
            index = 15
            res = [0] * int(length)
            flags = [0] * int(length)
            for line in fr.readlines():
                line = line.strip()  # 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                listFromLine = line.split('\t')  # \t表示空四个字符，也称缩进，相当于按一下Tab键
                four = int(listFromLine[4])
                five = int(listFromLine[0])
                if (four != 0 and flags[five - 1] == 0):
                    dict[five] = four
                if (four == 0 and len(dict) != 0):
                    # print(dict)
                    for key, value in dict.items():
                        if (res[int(key) - 1] == 0 and res[int(value) - 1] == 0):
                            res[int(key) - 1] = index
                            res[int(value) - 1] = index
                            flags[int(key) - 1] = 1
                            flags[int(value) - 1] = 1
                    index = index+10
                    dict.clear()
            result.append(res)
        return result,B


# 返回 RNA 二级结构二维数组
def helper5(length, width, arr=[]):
    matrix = [([255] * length) for i in range(width)]
    q = Queue(maxsize=0)  # 创建队列
    for x in arr:
        q.put(x)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (q.qsize() == 0):
                break
            matrix[i][j] = q.get()
    return matrix


def helper51(length, width, list=[]): #正反
    matrix = [([255] * length) for i in range(width)]
    # print(matrix)
    q = Queue(maxsize=0)  # 创建队列
    list = list[0:(length * width)]  # 保留length*width个元素
    # 编码RNA序列
    # A->0，G->85，C->170，U->255
    list_temp1 = []
    list_temp2 = []
    res = []
    for i in range(width):
        if i % 2 == 0:  # 偶数
            res = res + list[i * width:(i + 1) * width]
        if i % 2 != 0:  # 奇数
            list_temp1 = list[i * width:(i + 1) * width]
            for i in range(0, len(list_temp1)):
                list_temp2.insert(0, list_temp1[i])
            res = res + list_temp2
            list_temp1.clear()
            list_temp2.clear()

    for x in res:
        q.put(x)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            # print(len(matrix[i]))
            if (q.qsize() == 0):
                break
            matrix[i][j] = q.get()
    return matrix



# 制作RNA二级结构图像
def helper6(img, index,save_path):
    img.flags.writeable = True  # 将数组改为读写模式
    pic = Image.fromarray(np.uint8(img))
    pic.save(save_path + str(index) + '-f.png')

def helper61(img, index,save_path):
    img.flags.writeable = True  # 将数组改为读写模式
    pic = Image.fromarray(np.uint8(img))
    pic.save(save_path + str(index) + '-fb.png')

def main():
    name = '5srna'
    data,rname = helper4("..//data//"+name+"//",name)
    index = 1
    for x,y in zip(data,rname):
        # print(index,len(x))
        img = helper5(24,24,x)
        img1 = helper51(24,24,x)
        helper6(np.array(img),y,"..//picture//"+name+"//str_photo//")
        helper61(np.array(img1),y,"..//picture//"+name+"//str_photo//")
        index = index + 1

if __name__ == "__main__":
    main()