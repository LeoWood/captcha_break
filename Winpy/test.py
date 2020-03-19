#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/3/18 16:20
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
import string
from captcha.image import ImageCaptcha
characters = '-' + string.digits + string.ascii_uppercase

def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', target_size, (255, 255, 255))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式

    return new_image


# c = Image.open(r'C:\Users\Administrator\Desktop\ValidateCode (1).jpg')
# # c = Image.open(r'C:\Users\Administrator\Desktop\b.jpg')
# # c.convert('RGB')
# c = pad_image(c,(192,64))
# # plt.show(d)
# image = to_tensor(c)
#
# # generator = ImageCaptcha(width=192, height=64)
# # a = generator.generate_image('ABSD')
# # image = to_tensor(a)
#
# model=torch.load(r'E:\LiuHuan\Projects\captcha_break\Winpy\ctc3.pth')
# model = model.cuda()
#
# output = model(image.unsqueeze(0).cuda())
# output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
# print('pred:', decode(output_argmax[0]))


if __name__ == '__main__':
    model = torch.load(r'E:\LiuHuan\Projects\captcha_break\Winpy\ctc_3_fonts_5000_180_50.pth')
    model = model.cuda()
    while True:
        path = input()
        a = Image.open(path)
        c = pad_image(a, (192, 64))
        image = to_tensor(c)
        output = model(image.unsqueeze(0).cuda())
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        print('pred:', decode(output_argmax[0]))