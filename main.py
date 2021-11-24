import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.SelfTest.st_common import b2a_hex, a2b_hex
import math
import struct


def add_to_16(text):
    if len(text.encode('utf-8')) % 16:
        add = 16 - (len(text.encode('utf-8')) % 16)
    else:
        add = 0
    text = text + ('\0' * add)
    return text.encode('utf-8')


# 加密函数
def encrypt(text):
    key = '9999999999999999'.encode('utf-8')
    iv = b'qqqqqqqqqqqqqqqq'
    text = add_to_16(text)
    cryptos = AES.new(key, AES.MODE_CBC, iv)
    cipher_text = cryptos.encrypt(text)
    return b2a_hex(cipher_text)


# 解密后，去掉补足的空格用strip() 去掉
def decrypt(text):
    key = '9999999999999999'.encode('utf-8')
    iv = b'qqqqqqqqqqqqqqqq'
    cryptos = AES.new(key, AES.MODE_CBC, iv)
    plain_text = cryptos.decrypt(a2b_hex(text))
    return bytes.decode(plain_text).rstrip('\0')


def reverse(number):
    if number % 2 == 0:
        return number + 1
    else:
        return number - 1


def hm_encode(bit8):
    # print('Enter the data bits')
    d = bit8
    data = list(d)
    data.reverse()
    c, ch, j, r, h = 0, 0, 0, 0, []

    while ((len(d) + r + 1) > (pow(2, r))):
        r = r + 1

    for i in range(0, (r + len(data))):
        p = (2 ** c)

        if (p == (i + 1)):
            h.append(0)
            c = c + 1

        else:
            h.append(int(data[j]))
            j = j + 1

    for parity in range(0, (len(h))):
        ph = (2 ** ch)
        if (ph == (parity + 1)):
            startIndex = ph - 1
            i = startIndex
            toXor = []

            while (i < len(h)):
                block = h[i:i + ph]
                toXor.extend(block)
                i += 2 * ph

            for z in range(1, len(toXor)):
                h[startIndex] = h[startIndex] ^ toXor[z]
            ch += 1

    h.reverse()
    # print('Hamming code generated would be:- ', end="")

    return str(int(''.join(map(str, h))))


def hm_decode(bit12):
    d = bit12
    data = list(d)
    data.reverse()
    c, ch, j, r, error, h, parity_list, h_copy = 0, 0, 0, 0, 0, [], [], []

    for k in range(0, len(data)):
        p = (2 ** c)
        h.append(int(data[k]))
        h_copy.append(data[k])
        if (p == (k + 1)):
            c = c + 1

    for parity in range(0, (len(h))):
        ph = (2 ** ch)
        if (ph == (parity + 1)):

            startIndex = ph - 1
            i = startIndex
            toXor = []

            while (i < len(h)):
                block = h[i:i + ph]
                toXor.extend(block)
                i += 2 * ph

            for z in range(1, len(toXor)):
                h[startIndex] = h[startIndex] ^ toXor[z]
            parity_list.append(h[parity])
            ch += 1
    parity_list.reverse()
    error = sum(int(parity_list) * (2 ** i) for i, parity_list in enumerate(parity_list[::-1]))

    if ((error) == 0):
        str1 = bit12
        # print('There is no error in the hamming code received')
        return (str1[::-1][2] + str1[::-1][4:7] + str1[::-1][8:])[::-1]
    elif ((error) >= len(h_copy)):
        str1 = bit12
        # print('Error cannot be detected')
        return (str1[::-1][2] + str1[::-1][4:7] + str1[::-1][8:])[::-1]

    else:
        #print('Error is in', error, 'bit')

        if (h_copy[error - 1] == '0'):
            h_copy[error - 1] = '1'

        elif (h_copy[error - 1] == '1'):
            h_copy[error - 1] = '0'
            # print('After correction hamming code is:- ')
        h_copy.reverse()
        str1 = ''.join(map(str, h_copy))
        return (str1[::-1][2] + str1[::-1][4:7] + str1[::-1][8:])[::-1]


def LSB_encode(list, bit, local):
    carrier = list[8 * local: 8 * local + 8]
    if sum(carrier) % 2 == 0:
        if (bit + carrier[-1]) % 2 == 0:
            pass
        else:
            for i in range(0, len(carrier)):
                carrier[i] = reverse(carrier[i])
    else:
        carrier[-1] = reverse(carrier[-1])
        if (bit + carrier[-1]) % 2 == 0:
            pass
        else:
            for i in range(0, len(carrier)):
                carrier[i] = reverse(carrier[i])
        pass
    for x in range(0, 8):
        list[8 * local + x] = carrier[x]
    return list


def LSB_decode(carrier):
    # if sum(carrier) % 2 == 0:
    #     if carrier[-1] % 2:
    #         return '1'
    #     else:
    #         return '0'
    # else:
    #     return '0'
    if carrier[-1] % 2:
        return '1'
    else:
        return '0'


def getbin(e):
    str = ''
    for x in e:
        str = str + '{:07b}'.format(x)
    return str


def tobin(e):
    bytes = b''
    for i in range(0, len(e), 8):
        bytes = bytes + struct.pack('B', int(e[i:i + 8], 2))
    return bytes


def write(bin, image, path, repeat=None, times=None, random=True):
    # bin = bin + "101010101010101010101010101010101"
    if repeat is not None and repeat > 1:
        _bin = ''
        for x in bin:
            for i in range(0, repeat):
                _bin = _bin + x
        bin = _bin
    if times is not None and times > 1:
        bin_ = ''
        for i in range(0, times):
            bin_ = bin_ + bin
        bin = bin_
    gray_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    list = []
    for x in gray_img.tolist():
        list.extend(x)
    index = 0
    if random is True:
        np.random.seed(12345)
        local = set(np.random.random_integers(0, 9600, 10000).tolist())
    else:
        local = range(0, 10000)
    for x in local:
        list = LSB_encode(list, int(bin[index]), x)
        index = index + 1
        if index == len(bin):
            break
    newlist = []
    for x in range(0, 240):
        newlist.append(list[320 * x:320 * x + 320])
    array = np.asarray(newlist)
    cv2.imwrite(path, array)


def readimge(gray_img, repaet=None, random=True):
    list = []
    for x in gray_img.tolist():
        list.extend(x)
    np.random.seed(12345)
    if random is True:
        local = set(np.random.random_integers(0, 9600, 10000).tolist())
    else:
        local = range(0, 10000)
    str = ''
    index = 0
    for x in local:
        str = str + LSB_decode(list[8 * x:8 * x + 8])
        index = index + 1
        if index > 6000:
            break

        # elif index > 40 and "101010101010101010101010101010101" in str:
        #     str = str.split("101010101010101010101010101010101")[0]
        #     break
    if repaet is not None and repaet > 1:
        _str = ''
        for i in range(0, int(len(str) / repaet)):
            if str[i * repaet:i * repaet + repaet].count("1") > str[i * repaet:i * repaet + repaet].count("0"):
                _str = _str + "1"
            else:
                _str = _str + "0"
        str = _str

    _str = ''
    for i in range(0, int(len(str) / 12)):
        _str = _str + hm_decode(str[i * 12:i * 12 + 12])
    str = _str
    with open("2.txt", "ab") as f:
        print(tobin(str))
        f.write(tobin(str))


def read(path, repaet=None, random=True):
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    readimge(gray_img, repaet, random)


def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out  # 这里也会返回噪声，注意返回值




def sp_noise(noise_img, proportion):
    '''
    添加椒盐噪声
    proportion的值表示加入噪声的量，可根据需要自行调整
    return: img_noise
    '''
    height, width = noise_img.shape[0], noise_img.shape[1]  # 获取高度宽度像素值
    num = int(height * width * proportion)  # 一个准备加入多少噪声小点
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def crop(image, proportion):
    img_noise = image
    rows, cols = img_noise.shape
    for i in range(0, int(proportion * rows)):
        img_noise[i,] = 255
    return img_noise


if __name__ == '__main__':
    with open("1.txt", 'rb') as f:
        str0 = f.read()
        s = ''
        for x in str0:
            _x = (bin(x)[2:].zfill(8))
            s = s + hm_encode(_x)

    write(s, "lena_gray.png", "result.png", 1, 1, False)
    try:
        os.remove("2.txt")
    except:
        pass
    gray_img = cv2.imread("result.png", cv2.IMREAD_GRAYSCALE)

    #readimge(gaussian_noise(gray_img, 0, 0.00000002), 5, False)
    readimge(sp_noise(gray_img, 0.1), 1,False)
    cv2.imwrite("sp.png",sp_noise(gray_img, 0.4))
    # readimge(crop(gray_img, 0.1),5, False)

    #read("result.png", 5, False)
