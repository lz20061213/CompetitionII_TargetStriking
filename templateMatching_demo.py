#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: templateMatching_demo.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 下午3:17
# --------------------------------------------------------

import cv2
import numpy as np
import re
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import time

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


# function to fit
def func(params, x):
    a, b = params
    return a/x + b

# error
def error(params, x, y):
    return func(params, x) - y

# solver
def slovePara(X, Y):
    p0 = [1, -1]
    Para = leastsq(error, p0, args=(X, Y))
    return Para

def height2scale_regression(record_path):
    x = []  # height
    y = []  # pixel_ratio
    records = open(record_path, 'r').readlines()
    records = records[1:-1]
    # get the heights and pixel_ratios
    for record in records:
        record = record.strip()
        record = record.replace('\t\t\t', '\t\t')
        record = record.replace('\t\t', '\t')
        words = re.split('\t', record)
        fly_height = float(words[0])
        blue_pixels = float(words[1])
        red_pixels = float(words[2])
        width = float(words[3])
        pixel_ratio = (blue_pixels + red_pixels * 7 / 2) / 2 / width
        x.append(fly_height)
        y.append(pixel_ratio)
    # do bi regression
    x = np.array(x)
    y = np.array(y)
    Para = slovePara(x, y)
    a, b= Para[0]
    print "a=",a," b=",b
    print "cost:" + str(Para[1])
    print "求解的曲线是:"
    print("y="+str(round(a,2))+"/x+"+str(round(b,2)))

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color="green", label="sample data", linewidth=2)

    #   画拟合直线
    x_fit = np.linspace(10, 100, 100) ##在0-15直接画100个连续点
    y_fit = a/x_fit+b ##函数式
    plt.plot(x_fit, y_fit, color="red", label="solution line", linewidth=2)
    plt.legend() #绘制图例
    plt.show()
    return Para[0]


def getScaleRatioFromHeight(h, minh=10, maxh=100, params=None):
    assert h >= minh and h<= maxh
    return func(params, h)

# gray image
def templateMatching(im, template, method=cv2.TM_CCOEFF_NORMED):
    result = cv2.matchTemplate(im, template, method)
    return result

# create template
def createTemplate(ratio=2/7, rmax=960):
    template = np.zeros(shape=(rmax*2, rmax*2, 3), dtype=np.uint8)
    center = (rmax, rmax)
    cv2.circle(template, center, rmax, (255, 0, 0), -1, 8)
    cv2.circle(template, center, rmax*2/7, (0, 0, 255), -1, 8)
    #cv2.imshow('template', template)
    #cv2.imwrite('images/circle.jpg', template)
    #cv2.waitKey(0)
    return template

def TemplateMatching(impath, height, im=None, matchingRMin=15.0,  height_var=8):
    assert (impath == None or im == None)
    if impath  is not None:
        im = cv2.imread(impath, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im = cv2.resize(im, None, None, 0.5, 0.5, cv2.INTER_CUBIC)
    #[imh, ims, imv] = cv2.split(im)
    im = im_gray
    #cv2.imshow('search', imv)
    #cv2.waitKey(0)
    params = [4.3465, 0]
    rect = None
    score = None
    start = time.time()
    for i in range(height-height_var, height+height_var):
        scaleRatio = getScaleRatioFromHeight(h=i, params=params)
        r = int(scaleRatio*im.shape[1]/2)
        template = createTemplate(rmax=r)
        #start_m = time.time()
        template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #end_m = time.time()
        #print 'BGR2Gray time cost: {}'.format(end_m-start_m)
        #[templateh, templates, templatev] = cv2.split(template)
        template = template_gray
        #cv2.imshow('template', template)
        #cv2.waitKey(0)

        # scale the template image
        if r > matchingRMin:
            scaleMatching = matchingRMin / r
            #print scaleMatching
            template = cv2.resize(template, None, None, scaleMatching, scaleMatching, cv2.INTER_CUBIC)
            im_matching = cv2.resize(im, None, None, scaleMatching, scaleMatching, cv2.INTER_CUBIC)

        method = cv2.TM_CCOEFF_NORMED
        result = templateMatching(im_matching, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        w, h = template.shape[::-1]
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            if score is None or score > min_val:
                score = min_val
                rect = [int(top_left[0] / scaleMatching + 0.5), int(top_left[1] / scaleMatching + 0.5),
                        int(bottom_right[0] / scaleMatching + 0.5), int(bottom_right[1] / scaleMatching + 0.5)]
        else:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            if score is None or score < max_val:
                score = max_val
                rect = [int(top_left[0] / scaleMatching + 0.5), int(top_left[1] / scaleMatching + 0.5),
                        int(bottom_right[0] / scaleMatching + 0.5), int(bottom_right[1] / scaleMatching + 0.5)]

    end = time.time()
    print 'Total time cost: {}'.format(end-start)
    #cv2.imshow('im', im)
    #cv2.imshow('template', template)
    #cv2.rectangle(im, (rect[0], rect[1]), (rect[2], rect[3]), 255, 2)
    #im = cv2.resize(im, None, None, 0.25, 0.25, cv2.INTER_CUBIC)
    #cv2.imshow('matching result', im)
    #cv2.waitKey(1)
    return rect

def extractSymbol(video_path, height, saveroot):
    videoCapture = cv2.VideoCapture(video_path)
    success, frame = videoCapture.read()
    count = 0
    while success:
        rect = TemplateMatching(None, height, frame)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 4)
        center_x = (rect[0] + rect[2]) / 2
        center_y = (rect[1] + rect[3]) / 2
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        target_x_min = int(center_x - w / 7 * 3 / 2)
        target_y_min = int(center_y - h / 7 * 3 / 2)
        target_x_max = int(center_x + w / 7 * 3 / 2)
        target_y_max = int(center_y + h / 7 * 3 / 2)
        symbol_im = frame[target_y_min:target_y_max, target_x_min:target_x_max]
        impath = saveroot + '{:0>6}.jpg'.format(count)
        cv2.imwrite(impath, symbol_im)
        count += 1
        # for showing
        #cv2.imshow('symbol', symbol_im)
        cv2.rectangle(frame, (target_x_min, target_y_min), (target_x_max, target_y_max), (0, 255, 0), 12)
        frame = cv2.resize(frame, None, None, 0.25, 0.25, cv2.INTER_CUBIC)
        cv2.imshow('matching result', frame)
        cv2.waitKey(1)
        success, frame = videoCapture.read()

def TestCreateTemplate():
    template = createTemplate()
    cv2.imshow('template', template)
    cv2.waitKey(0)

def TestRegression():
    height2scale_regression('records/height_circle_mapping.txt')

def TestTemplateMatching():
    record_path = 'records/height_circle_mapping.txt'
    records = open(record_path, 'r').readlines()
    records = records[1:-1]
    for record in records:
        record = record.strip()
        record = record.replace('\t\t\t', '\t\t')
        record = record.replace('\t\t', '\t')
        words = re.split('\t', record)
        id = int(words[5])
        height = float(words[0])
        TemplateMatching('records/height_circle/DJI_{:0>4}.jpg'.format(id), height)

def TestExtractSymbol():
    pass

if __name__ == '__main__':
    # TestCreateTemplate()
    #TestRegression()
    # TestTemplateMatching()
    extractSymbol('F:/uva/Mavic-20180509-天津海教园/DJI_0053.MP4', 50, 'records/symbol_T/50/')
