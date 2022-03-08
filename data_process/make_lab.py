from numpy import *
import numpy as np
import os
from numpy import *
import os
import imghdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from queue import Queue, LifoQueue, PriorityQueue

# 第一种方法
# index = 1
# with open("label_test.txt", "r") as f:
#     for line in f.readlines():
#         word = line.split("-")
#         a = word[0]
#         b = word[1]
#         c = word[new]
#         num = a.split("/")[new]
#         print(index)
#         index = index + 1
#         with open("label_3.txt", "a") as file:
#             file.write("SeqImg_2/" + str(num) + "-" + "StrImg/"+ num + "-" + c)
#
# f.close()  # 关闭文件

lable = {'5s':1,'16s':2,'23s':3,'grp1':4,'grp2':5,'RNaseP':6,'srp':7,'tmRNA':8,'tRNA':9,'telomerase':10}
# 基础标签
def Read_File(path,data_path):
    files = os.listdir(data_path)
    with open('../lable/'+path+'/'+path+'-De-redundancy.txt','r') as fd:
        for lines in fd.readlines():
            line=lines.strip()
            if line in files:
                fname = data_path + line
                with open(fname,'r') as fr:
                    lines = fr.readlines()
                    first_line = lines[0]
                    first_line = first_line.split('\t')
                with open("../lable/"+path+'/'+path+'.txt','a') as f:
                    for key, value in lable.items():
                        if key in line:
                            f.write(line + '*' + first_line[0]+'*'+str(value)+'\n')



#  数据标签
def Read_File2(path,data_path):
    files = os.listdir(data_path)
    with open('../lable/'+path+'/'+path+'-De-redundancy.txt','r') as fd:
        for lines in fd.readlines():
            line=lines.strip() #去冗余名字
            if line in files:
                fname = data_path + line
                with open(fname,'r') as fr:
                    lines = fr.readlines()
                    first_line = lines[0]
                    first_line = first_line.split('\t') #取出对应去冗余数据的序列长度
                with open("../lable/"+path+'/'+path+'-data.txt','a') as f:
                    line = line.rstrip('.ct')
                    for key, value in lable.items():
                        if key in line:
                            f.write("../picture/"+path+"/seq_photo/" +str(line) +'-f.png'+ "*" + "../picture/"+path+"/str_photo/" + str(line)+ "-f.png" + "*" + first_line[0]+ "*" +str(value)+ '\n')
                            f.write("../picture/"+path+"/seq_photo/" +str(line) +'-fb.png'+ "*" + "../picture/"+path+"/str_photo/" + str(line)+ "-fb.png" + "*" + first_line[0]+ "*" +str(value)+ '\n')
                            f.write("../picture/"+path+"/seq_photo/" +str(line) +'-f-l-r.png'+ "*" + "../picture/"+path+"/str_photo/" + str(line)+ "-f-l-r.png" + "*" + first_line[0]+ "*" +str(value)+ '\n')
                            f.write("../picture/"+path+"/seq_photo/" +str(line) +'-fb-l-r.png'+ "*" + "../picture/"+path+"/str_photo/" + str(line)+ "-fb-l-r.png" + "*" + first_line[0]+ "*" +str(value)+ '\n')
                            f.write("../picture/"+path+"/seq_photo/" +str(line) +'-f-u-d.png'+ "*" + "../picture/"+path+"/str_photo/" + str(line)+ "-f-u-d.png" + "*" + first_line[0]+ "*" +str(value)+'\n')
                            f.write("../picture/"+path+"/seq_photo/" +str(line) +'-fb-u-d.png'+ "*" + "../picture/"+path+"/str_photo/" + str(line)+ "-fb-u-d.png" + "*" + first_line[0]+ "*" +str(value)+'\n')


name = 'tmrna'
datasets = Read_File(name,"/home/yuanshuai20/RNAs/data/"+name+"/")
datasets = Read_File2(name,"/home/yuanshuai20/RNAs/data/" + name + "/")