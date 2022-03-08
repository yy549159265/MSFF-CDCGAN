# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:51:06 2018

@author: zhangch
"""

from __future__ import division
import os
import csv
import shutil

def readfile(path_dir,file):
    filename = path_dir + file
    csvFile = open(filename,"r")
    reader = csv.reader(csvFile)
    data = []
    for item in reader:
        data.append(item)
    return data,file

def distinct(list_dis,string):
    flag = False
    if string not in list_dis:
        flage = True
        for i in list_dis:
            if string in i:
                flage = False
        if flage == True:
            list_dis.append(string)
            flag = True
    return flag,list_dis

path1 = 'data_set'
path_dir = '/home/yuanshuai20/RNAs/lable/'+path1+'/' + path1 + '-De-redundancy/'
save_dir = '/home/yuanshuai20/RNAs/lable/'+path1+'/'
if __name__ == "__main__":
    path = os.listdir(path_dir)
    dict1 = {}
    for i in path:
        data,filename = readfile(path_dir , i)
        # print(filename)
        dict1[filename] = len(data)
    dict2 = sorted(dict1.items(),key=lambda d:d[1],reverse=True)
    pathdir = []
    for i in range(len(dict2)):
        pathdir.append(dict2[i][0])
    list_dis = []
    for i in pathdir:
        data,filename = readfile(path_dir , i)
        string = ''
        for i in range(len(data)):
            string = string + data[i][0]
        flag,list_dis = distinct(list_dis,string)
        if flag == True:
            filename = filename.rstrip('.csv')
            with open('/home/yuanshuai20/RNAs/lable/'+path1+'/'+path1+'-De-redundancy.txt','a')as f:
                f.write(filename+'.ct'+'\n')

            # shutil.copyfile(path_dir + filename , save_dir + filename)

