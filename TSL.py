# coding:utf-8
from PIL import Image
import numpy as np
import math
import time
import os  # 用于查找目录下的文件
import copy
import sys
import collections
from skimage.measure import compare_ssim
import cv2
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
import random


# 输出图片的位置
ImageWidth = 512
ImageHeight = 512
FILE_PATH = r"" % (ImageWidth, ImageHeight)
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)


def SaveResult(str):

    try:
        fname = time.strftime("%Y%m%d", time.localtime())
        f2 = open(FILE_PATH + ' ' + fname + '.txt', 'a+')
        f2.read()

        f2.write('\n')
        f2.write(str)
        f2.write('\n')
    finally:
        if f2:
            f2.close()
    return 0


def PSNR(image_array1, image_array2):
    assert (np.size(image_array1) == np.size(image_array2))
    n = np.size(image_array1)
    assert (n > 0)
    MSE = 0.0
    for i in range(0, n):
        MSE += math.pow(int(image_array1[i]) - int(image_array2[i]), 2)
    MSE = MSE / n
    if MSE > 0:
        rtnPSNR = 10 * math.log10(255 * 255 / MSE)
    else:
        rtnPSNR = 100
    return rtnPSNR


def dec2bin_higher_ahead(x, n):
    b_array1 = np.zeros(n)
    for i in range(0, n, 1):
        b_array1[i] = int(x % 2)
        x = x // 2
    b_array2 = np.zeros(n)
    for i in range(0, n, 1):
        b_array2[i] = b_array1[n - i - 1]  # n-1-i ？
    return b_array2

def dec2bin_lower_ahead(y, n):
    x = y
    b_array1 = np.zeros(n)
    for i in range(0, n, 1):
        b_array1[i] = int(x % 2)
        x = x // 2

    return b_array1

def binToDec(binary):
    result = 0
    for i in range(len(binary)):
        result += int(binary[-(i + 1)]) * pow(2, i)
    return result

# ALGORITHM: SS 方法
def TSL2022(image_array, secret_string, n=4,k = 2, image_file_name=''):
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups, n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0, n):
            if (i * n + j < image_array.size):
                pixels_group[i, j] = image_array[i * n + j]
        i = i + 1
    embedded_pixels_group = pixels_group.copy()

    num = n + n * k
    num_secret_groups = math.ceil(secret_string.size / num)
    secret_group = np.zeros((num_secret_groups, num))
    i = 0
    while (i < num_secret_groups):
        for j in range(0, num):
            if (i * num + j < s_data.size):
                secret_group[i, j] = s_data[i * num + j]
        i = i + 1
    secret_group_first = np.zeros((num_secret_groups, n))
    secret_group_last = np.zeros((num_secret_groups, k * n))
    i = 0
    while (i < num_secret_groups):
        for j in range(0, n):
            secret_group_first[i] = secret_group[i][0:n]
        i = i + 1
    i = 0
    while (i < num_secret_groups):
        for j in range(0, k * n):
            secret_group_last[i] = secret_group[i][-(k * n):]
        i = i + 1
    pixels_group_temp = np.zeros((num_secret_groups, n))
    for i in range(0, num_secret_groups):
        for j in range(0, n):
            pixels_group_temp[i][j] = math.floor(pixels_group[i][j] / (2**k))
            if pixels_group_temp[i][j]==0:
                pixels_group_temp[i][j] = pixels_group_temp[i][j] + 1
            if (pixels_group_temp[i][j] + 1) * (2 * k) >= 255:
                pixels_group_temp[i][j] = pixels_group_temp[i][j] - 1

    def dec2bin_lower_ahead(x, n):
        b_array1 = np.zeros(n + 1)
        for i in range(0, n + 1, 1):
            b_array1[i] = int(x % 2)
            x = x // 2
        return b_array1[0],b_array1[1]
    ABCD_arr = np.zeros((num_secret_groups, n * 2))
    for i in range(0, num_secret_groups):
        for j in range(0, n):
            A = np.zeros(2)
            A1,A2 = dec2bin_lower_ahead(pixels_group_temp[i][j], 8)
            A[0] += A1
            A[1] += A2
            for p in range(0, 2):
                ABCD_arr[i][j * 2 + p] += A[p]

    if (n==2):
        ABCD_arr_1 = np.zeros((num_secret_groups, n))
        for i in range(0, num_secret_groups):
            num_A = collections.Counter(ABCD_arr[i][0: 3])[1]
            num_B = collections.Counter(ABCD_arr[i][2: 4])[1] + int(ABCD_arr[i][0])
            if (num_A % 2) == 0:
                A = 0
            else:
                A = 1
            if (num_B % 2) == 0:
                B = 0
            else:
                B = 1
        for i in range(0, num_secret_groups):
            ABCD_arr_1[i][0] = A
            ABCD_arr_1[i][1] = B
        for i in range(0, num_secret_groups):
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])):
                if (pixels_group_temp[i,1] % 2) == 0:
                    pixels_group_temp[i, 1] += 1
                else:
                    pixels_group_temp[i, 1] -= 1
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] -= 1
                else:
                    pixels_group_temp[i, 1] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])):
                # print('进来这一步了吗？')
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] -= 1
                else:
                    pixels_group_temp[i, 0] += 1
    if (n==3):
        ABCD_arr_1 = np.zeros((num_secret_groups, n))
        for i in range(0, num_secret_groups):
            num_A = collections.Counter(ABCD_arr[i][0: 3])[1]
            num_B = collections.Counter(ABCD_arr[i][2: 5])[1]
            num_C = collections.Counter(ABCD_arr[i][4: 6])[1] + int(ABCD_arr[i][0])
            if (num_A % 2) == 0:
                A = 0
            else:
                A = 1
            if (num_B % 2) == 0:
                B = 0
            else:
                B = 1
            if (num_C % 2) == 0:
                C = 0
            else:
                C = 1
        for i in range(0, num_secret_groups):
            ABCD_arr_1[i][0] = A
            ABCD_arr_1[i][1] = B
            ABCD_arr_1[i][2] = C
        for i in range(0, num_secret_groups):
            #----------------------------------------
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] -= 1
                else:
                    pixels_group_temp[i, 1] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2])):
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] -= 1
                else:
                    pixels_group_temp[i, 2] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2])):
                # print('进来这一步了吗？')
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] -= 1
                else:
                    pixels_group_temp[i, 0] += 1
            #----------------------------------------
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] += 1
                else:
                    pixels_group_temp[i, 1] -= 1
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] += 1
                else:
                    pixels_group_temp[i, 0] -= 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2])):
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] += 1
                else:
                    pixels_group_temp[i, 2] -= 1
            #----------------------------------------
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] -= 1
                else:
                    pixels_group_temp[i, 1] += 1
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] += 1
                else:
                    pixels_group_temp[i, 2] -= 1
    if (n==4):
        ABCD_arr_1 = np.zeros((num_secret_groups, n))
        for i in range(0, num_secret_groups):
            num_A = collections.Counter(ABCD_arr[i][0: 3])[1]
            num_B = collections.Counter(ABCD_arr[i][2: 5])[1]
            num_C = collections.Counter(ABCD_arr[i][4: 7])[1]
            num_D = collections.Counter(ABCD_arr[i][6: 8])[1] + int(ABCD_arr[i][0])
            if (num_A % 2) == 0:
                A = 0
            else:
                A = 1
            if (num_B % 2) == 0:
                B = 0
            else:
                B = 1
            if (num_C % 2) == 0:
                C = 0
            else:
                C = 1
            if (num_D % 2) == 0:
                D = 0
            else:
                D = 1
        for i in range(0, num_secret_groups):
            ABCD_arr_1[i][0] = A
            ABCD_arr_1[i][1] = B
            ABCD_arr_1[i][2] = C
            ABCD_arr_1[i][3] = D
        for i in range(0, num_secret_groups):
            #----------------------------------------
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i,1] % 2) == 0:
                    pixels_group_temp[i, 1] -= 1
                else:
                    pixels_group_temp[i, 1] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i,2] % 2) == 0:
                    pixels_group_temp[i, 2] -= 1
                else:
                    pixels_group_temp[i, 2] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i,3] % 2) == 0:
                    pixels_group_temp[i, 3] -= 1
                else:
                    pixels_group_temp[i, 3] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] -= 1
                else:
                    pixels_group_temp[i, 0] += 1
            # ----------------------------------------
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] += 1
                else:
                    pixels_group_temp[i, 1] -= 1
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] -= 1
                else:
                    pixels_group_temp[i, 1] += 1
                if (pixels_group_temp[i, 3] % 2) == 0:
                    pixels_group_temp[i, 3] -= 1
                else:
                    pixels_group_temp[i, 3] += 1
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] += 1
                else:
                    pixels_group_temp[i, 0] -= 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] += 1
                else:
                    pixels_group_temp[i, 2] -= 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] -= 1
                else:
                    pixels_group_temp[i, 0] += 1
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] -= 1
                else:
                    pixels_group_temp[i, 2] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 3] % 2) == 0:
                    pixels_group_temp[i, 3] += 1
                else:
                    pixels_group_temp[i, 3] -= 1
            # ----------------------------------------
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] == secret_group_first[i,3])):
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] -= 1
                else:
                    pixels_group_temp[i, 1] += 1
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] += 1
                else:
                    pixels_group_temp[i, 2] -= 1
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] == secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    # print('修改正确了吗')
                    pixels_group_temp[i, 0] -= 1
                else:
                    pixels_group_temp[i, 0] -= 1
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] += 1
                else:
                    pixels_group_temp[i, 1] -= 1
            if ((ABCD_arr_1[i,0] != secret_group_first[i, 0]) & (ABCD_arr_1[i,1] == secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] += 1
                else:
                    pixels_group_temp[i, 0] -= 1
                if (pixels_group_temp[i, 3] % 2) == 0:
                    pixels_group_temp[i, 3] -= 1
                else:
                    pixels_group_temp[i, 3] += 1
            if ((ABCD_arr_1[i,0] == secret_group_first[i, 0]) & (ABCD_arr_1[i,1] != secret_group_first[i,1])
                    & (ABCD_arr_1[i,2] != secret_group_first[i,2]) & (ABCD_arr_1[i,3] != secret_group_first[i,3])):
                if (pixels_group_temp[i, 2] % 2) == 0:
                    pixels_group_temp[i, 2] -= 1
                else:
                    pixels_group_temp[i, 2] += 1
                if (pixels_group_temp[i, 3] % 2) == 0:
                    pixels_group_temp[i, 3] += 1
                else:
                    pixels_group_temp[i, 3] -= 1
            # ----------------------------------------
            if ((ABCD_arr_1[i, 0] != secret_group_first[i, 0]) & (ABCD_arr_1[i, 1] != secret_group_first[i, 1])
                    & (ABCD_arr_1[i, 2] != secret_group_first[i, 2]) & (ABCD_arr_1[i, 3] != secret_group_first[i, 3])):
                if (pixels_group_temp[i, 0] % 2) == 0:
                    pixels_group_temp[i, 0] -= 1
                else:
                    pixels_group_temp[i, 0] += 1
                if (pixels_group_temp[i, 1] % 2) == 0:
                    pixels_group_temp[i, 1] += 1
                else:
                    pixels_group_temp[i, 1] -= 1
                if (pixels_group_temp[i, 3] % 2) == 0:
                    pixels_group_temp[i, 3] -= 1
                else:
                    pixels_group_temp[i, 3] += 1

    for i in range(0, num_secret_groups):
        for j in range(0,n):
            pixels_group_temp[i][j] *= (2 ** k)
            for p in range(k-1, -1, -1):
                pixels_group_temp[i][j] += secret_group_last[i][j * k + p] * (2 ** (k - 1 - p))

    for i in range(0,num_secret_groups):
        for j in range(0,n):
            embedded_pixels_group[i][j] = pixels_group_temp[i][j]

    num_pixels_changed = num_secret_groups * n
    img_out1 = embedded_pixels_group.flatten()
    img_out2 = img_out1[:ImageWidth * ImageHeight]
    img_array_out = img_out2.copy()
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)

    img_out = img_out2.reshape(ImageWidth, ImageHeight)
    img_out = Image.fromarray(img_out)
    img_out = img_out.convert('L')

    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = FILE_PATH + '\\' + originfilename + '_' + sys._getframe().f_code.co_name + "_n_" + str(n) + "_k_" + str(
        k) + ".png"
    img_out.save(new_file, 'png')

    img1 = image_array.reshape(ImageWidth, ImageHeight)
    img2 = np.array(Image.open(new_file))

    plt.hist([image_array, img_out])
    plt.show()

    str1 = 'Image:%30s,Method:%15s,n=%d,k=%d,pixels used: %d,PSNR: %.2f,SSIM: %.4f,QI: %.4f' % (
        originfilename, sys._getframe().f_code.co_name, n, k, num_pixels_changed, psnr,ssim(img1, img2),QI)
    print(str1)
    SaveResult('\n' + str1)

    return 0


def proof():
    n = 2
    k = 5
    moshu = 2 ** (n * k + 1)
    c0 = 3
    c1 = 11
    outlist = []
    for g0 in range(0, 256):
        for g1 in range(0, 256):
            d = (c0 * g0 + c1 * g1) % moshu
            if d not in outlist:
                outlist.append(d)
    outlist.sort()
    assert (len(outlist) == moshu)
    return


for file in os.listdir(path):
    file_path = os.path.join(path, file)
    # if "Pepper.png" not in file_path:
    #    continue
    # if "Tiffany.png" not in file_path:
    #    continue
    if os.path.isfile(file_path):
        print(file_path)
        img = Image.open(file_path, "r")
        img = img.convert('L')

        img_array1 = np.array(img)
        img_array2 = img_array1.reshape(img_array1.shape[0] * img_array1.shape[1])
        img_array3 = img_array1.flatten()


        TSL2022(img_array3, s_data, 2, 1, file_path)

        print('-----------------------')
SaveResult('end')
time.sleep(10)