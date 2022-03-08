#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 16 13:36:01 2018

@author: zhangch
"""
import os
import pandas as pd
import csv


# 遍历指定目录，显示目录下的所有文件名

def readfile(fileDir, filename):
    fopen = open(fileDir + filename, 'r')
    rnafile = fopen.readlines()
    del rnafile[0]
    with open("test.txt", 'w') as f:
        for i in rnafile:
            f.write(i)
    data = pd.read_csv("test.txt", sep='\t', header=None)
    return data, filename


def transform(data, filename):
    rnaseq = data.loc[:, 1]
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] == 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return rnaseq, rnastructure, filename


def savefile(rnaseq, rnastructure, filename):
    rnaseqfile = seq_Dir + filename[:-3] + ".csv"
    rnaseqcsv = open(rnaseqfile, 'w')
    seqwriter = csv.writer(rnaseqcsv)
    m = len(rnastructure)
    for i in range(m):
        seqwriter.writerow(rnaseq[i])


path = 'data_set'
list_Dir = '/home/yuanshuai20/RNAs/data/' + path + '/'
seq_Dir = '/home/yuanshuai20/RNAs/lable/'+path+'/' + path + '-De-redundancy/'

if __name__ == '__main__':
    pathDir = os.listdir(list_Dir)
    for i in pathDir:
        data, filename = readfile(list_Dir, i)
        rnaseq, rnastructure, filename = transform(data, filename)
        savefile(rnaseq, rnastructure, filename)