from PIL import Image
import numpy as np
import torch.nn as nn
import torch
from itertools import groupby
import random
# 辅助统计精度1
def helper1(index,array):
    row = int(index%60)
    col = index%30
    # print('row:',row,'col:',col)
    return array[row][col]
# 辅助统计精度2
def error(RNA_len,real_img,fake_img):
    nums = 0
    for index in range(RNA_len):
        num1 = helper1(index,real_img)
        num2 = helper1(index,fake_img)
        if(num1.all()==num2.all()):
            nums=nums+1
    return nums

def Correction_unit(fake_str_path,name,RNAlen,d):
    if 'f.png' in name:
        img_fake = Image.open(fake_str_path) #图像读取
        img_fake = np.array(img_fake.convert('L')) #变成numpy
        con = img_fake.flatten()
        con = Correction(con,RNAlen)
        con = np.array(np.split(con, d))
        con = con.flatten()
        return con
    elif 'f-l-r.png' in name:
        img_fake = Image.open(fake_str_path)
        img_fake = img_fake.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        con = img_fake.flatten()
        con = Correction(con,RNAlen)
        con = np.array(np.split(con, d))
        img_fake = Image.fromarray(con, 'L')
        img_fake = img_fake.transpose(Image.FLIP_LEFT_RIGHT)
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        con = img_fake.flatten()
        return con
    elif 'f-u-d.png' in name:
        img_fake = Image.open(fake_str_path)
        img_fake = img_fake.transpose(Image.FLIP_TOP_BOTTOM) #上下翻转
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        con = img_fake.flatten()
        con = Correction(con,RNAlen)
        con = np.array(np.split(con, d))
        img_fake = Image.fromarray(con, 'L')
        img_fake = img_fake.transpose(Image.FLIP_TOP_BOTTOM) #上下翻转
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        con = img_fake.flatten()
        return con
    elif 'fb.png' in name:
        img_fake = Image.open(fake_str_path) #图像读取
        img_fake = np.array(img_fake.convert('L')) #变成numpy
        img_fake = transformation_f_fb(img_fake) #fb->f
        con = img_fake.flatten() #fb->b
        con = Correction(con,RNAlen)
        con = np.array(np.split(con, d))
        con = transformation_f_fb(con)  #f->fb
        con = con.flatten()
        return con
    elif 'fb-l-r.png' in name:
        img_fake = Image.open(fake_str_path)
        img_fake = img_fake.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        img_fake = transformation_f_fb(img_fake)
        con = img_fake.flatten()
        con = Correction(con,RNAlen)
        con = np.array(np.split(con, d))
        img_fake = transformation_f_fb(con)
        img_fake = Image.fromarray(img_fake, 'L')
        img_fake = img_fake.transpose(Image.FLIP_LEFT_RIGHT)
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        con = img_fake.flatten()
        return con
    elif 'fb-u-d.png' in name:
        img_fake = Image.open(fake_str_path)
        img_fake = img_fake.transpose(Image.FLIP_TOP_BOTTOM) #上下翻转
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        img_fake = transformation_f_fb(img_fake)
        con = img_fake.flatten()
        con = Correction(con,RNAlen)
        con = np.array(np.split(con, d))
        img_fake = transformation_f_fb(con)
        img_fake = Image.fromarray(img_fake, 'L')
        img_fake = img_fake.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
        img_fake = np.array(img_fake.convert('L'))  # 变成numpy
        con = img_fake.flatten()
        return con

def transformation_f_fb(img_fake):
    for i, j in enumerate(img_fake):  # 0 [1 1 1 1 1 1 1 0 0 2 2 2 0 0 0 0]
        if i % 2 != 0:  # 奇数
            n = 0
            m = len(j) - 1
            for a in range(len(j)):
                if j[m] == 255:
                    m = m - 1
            for b in range(len(j)):
                if n < m:
                    temp = j[n]
                    j[n] = j[m]
                    j[m] = temp
                    n = n + 1
                    m = m - 1
    return img_fake

def Correction(con,RNAlen):
    for i,j in enumerate(con):
        if i<RNAlen:
            if j<10:
                con[i] = 0
            else:
                b = j//10%10
                c = j//100%10
                con[i] = c*100+b*10+5

    return con

###### 真假结构矩阵元素获取
def Structure_matrix(real_path, fake_path,RNA_len):
    img_fake = Image.open(fake_path)
    img_real = Image.open(real_path)
    # 转换图片的模式为PIL转np转tensor
    img_fake = np.array(img_fake.convert('L'))
    img_real = np.array(img_real.convert('L'))
    # img_src = torch.from_numpy(img_src)
    img_fake = img_fake.flatten()
    img_real = img_real.flatten()

    return img_real,int(RNA_len )


###### 计算R,P,F1,ACC
def calculation(real,fake,RNALEN):

    # TP表示在RNA中真实存在且被正确预测出的配对碱基的个数
    TP = 0
    # TN表示在RNA中真实存在且被正确预测出的非配对碱基的个数
    TN = 0
    # FN表示在RNA中真实存在且没有正确预测出的配对碱基的个数
    FN = 0
    # FP表示在RNA中根本不存在且被错误预测出的非配对碱基的个数
    FP = 0
    epsilon = 1e-7
    a, b, c = real,fake,RNALEN  # a是真结构矩阵，b是假结构矩阵，c是RNA长度
    for x, y in enumerate(zip(a, b)):
        # x 索引 y[0]是真结构 y[1]假结构
        # print('x:', x,'y0:',y[0],'y1:',y[1])

        if x < c:
            if y[0] ==y[1]:
                if y[0] ==0 and y[1] ==0:
                    TN = TN + 1
                if y[0] != 0 and y[1] != 0:
                    TP = TP + 1
            if y[0] != y[1]:
                if  y[0] !=0 and y[1] ==0 :
                    FN = FN + 1
                if y[0] == 0 and y[1] !=0:
                    FP = FP + 1
        else:
            break
        #     if y[1] == 0:
        #         if y[1] == y[0]:
        #             TN = TN + 1 # 预测与实际一样且为非配对
        #         else:
        #             FN = FN + 1 # 预测与实际不一样且为非配对
        #     else:
        #         if y[1] == y[0]:
        #             TP = TP + 1 # 预测与实际一样且为配对
        #         else:
        #             if y[1] not in a:
        #                 FP = FP + 1 # 预测与实际不一样且为配对
        # else:
        #     break

    # R = TP / (TP + FN+epsilon)
    # P = TP / (TP + FP+epsilon)
    # F1 = (2 * R*P) / (R+P+epsilon)
    Accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    TPR = float(TP) / float(TP+FN+epsilon)
    FPR = float(FP) / float(FP+TN+epsilon)
    return TP,TN,FP,FN,Accuracy,TPR,FPR

###### gp评价指标
def gradient_penalty(D, xr, xf, c,batch_size):
    """


    :param D:
    :param xr:f
    :param xf:
    :return:
    """
    LAMBDA = 0.3

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, new]
    alpha = torch.rand(batch_size, 1, 60, 60).cuda()
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(c,interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp





# Clamp函数x限制在区间[min, max]内
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


