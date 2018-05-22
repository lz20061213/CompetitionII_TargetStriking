#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: symbolRecognizing_demo.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 13:48
# --------------------------------------------------------

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

def load_data(root_path, imsize=(64, 64), method='pixel'):
    # F: 0, T: 1
    images = []
    samples = []
    responses = []
    symbol_paths = ['symbol_T', 'symbol_F']
    heights = ['20', '30', '50']
    for symbol_path in symbol_paths:
        for height in heights:
            img_folder = os.path.join(root_path, symbol_path, height)
            if 'F' in symbol_path:
                response = 0
            else:
                response = 1
            for imgname in os.listdir(img_folder):
                imgpath = os.path.join(img_folder, imgname)
                im = cv2.imread(imgpath)
                im = cv2.resize(im, imsize, cv2.INTER_CUBIC)
                images.append(im)
                # cv2.imshow('sample', im)
                # cv2.waitKey(0)
                if method is 'pixel':
                    sample = im.flatten()
                elif method is 'hog':
                    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    sample = extract_hog(im)
                samples.append(sample)
                responses.append(response)
    images = np.array(images)
    samples = np.array(samples).astype(np.float32)
    responses = np.array(responses).astype(np.int32)
    indexs = range(samples.shape[0])
    random.shuffle(indexs)
    images = images[indexs, :]
    samples = samples[indexs, :]
    responses = responses[indexs]
    return images, samples, responses

def extract_hog(im, winSize=(64,64), blockSize=(16,16), blockStride=(8, 8), cellSize=(8,8)):
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, 9)
    descriptor = hog.compute(im)
    if descriptor is None:
        assert 'hog features is None'
    else:
        descriptor = descriptor.ravel()
    return descriptor

def extract_lbp(im):
    pass

class Symbol(object):
    model = None
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

class SVM(Symbol):
    ratio = 0.8
    def __init__(self):
        self.model = LinearSVC()

    def load(self, fn):
        r = np.load(fn)
        classes_ = r['classes']
        coef_ = r['coef']
        intercept_ = r['intercept']
        self.model.__setattr__('classes_', classes_)
        self.model.__setattr__('coef_', coef_)
        self.model.__setattr__('intercept_', intercept_)


    def train(self, samples, responses):
       self.model.fit(samples, responses)

    def predict(self, samples):
        return self.model.predict(samples)

    def save(self, path):
        classes_ = self.model.classes_
        coef_ = self.model.coef_
        intercept_ = self.model.intercept_
        np.savez(path, classes=classes_, coef=coef_, intercept=intercept_)

def train_symbols():
    print 'loading data ...'
    images, samples, responses = load_data(root_path='./records', method='hog')
    model = SVM()
    train_n = int(samples.shape[0] * model.ratio)
    print 'training SVM ...'
    model.train(samples[:train_n, :], responses[:train_n])
    model.save(path='./records/coeffs.npz')
    train_rate = np.mean(model.predict(samples[:train_n, :]) == responses[:train_n]) #前一半进行训练，并得到训练准确率
    test_rate = np.mean(model.predict(samples[train_n:, :]) == responses[train_n:]) #后一半进行测试，并得到测试准确率
    print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)

def test_symbols():
    model = SVM()
    print 'loading coeffs ...'
    model.load(fn='./records/coeffs.npz')
    '''
    image = cv2.imread('images/pimg.jpg')
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
    feature = extract_hog(image)
    decision = model.predict(feature.reshape(1, -1))[0]
    print decision
    '''
    print 'loading data ...'
    images, samples, responses = load_data(root_path='./records', method='hog')
    for image, sample, response in zip(images, samples, responses):
        decision = model.predict(sample.reshape(1, -1))[0]
        image = cv2.resize(image, None, None, 2.5, 2.5, cv2.INTER_CUBIC)
        cv2.putText(image, 'GT:{}'.format(response), (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, 'PT:{}'.format(decision), (5, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('predict', image)
        cv2.waitKey(0)


def train_points():
    # 训练的点数
    train_pts = 30

    # 创建测试的数据点，2类
    # 以(-1.5, -1.5)为中心
    rand1 = np.ones((train_pts,2)) * (-2) + np.random.rand(train_pts, 2)
    print('rand1：')
    print(rand1)

    # 以(1.5, 1.5)为中心
    rand2 = np.ones((train_pts,2)) + np.random.rand(train_pts, 2)
    print('rand2:')
    print(rand2)

    # 合并随机点，得到训练数据
    train_data = np.vstack((rand1, rand2))
    train_data = np.array(train_data, dtype='float32')
    train_label = np.vstack( (np.zeros((train_pts,1), dtype='int32'), np.ones((train_pts,1), dtype='int32')))

    # 显示训练数据
    plt.figure(1)
    plt.plot(rand1[:,0], rand1[:,1], 'o')
    plt.plot(rand2[:,0], rand2[:,1], 'o')
    plt.plot(rand2[:,0], rand2[:,1], 'o')

    # 创建分类器
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)  # SVM类型
    svm.setKernel(cv2.ml.SVM_LINEAR) # 使用线性核
    svm.setC(1.0)

    # 训练
    ret = svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)

    # 测试数据，20个点[-2,2]
    pt = np.array(np.random.rand(20,2) * 4 - 2, dtype='float32')
    (ret, res) = svm.predict(pt)
    print("res = ")
    print(res)

    # 按label进行分类显示
    plt.figure(2)
    res = np.hstack((res, res))

    # 第一类
    type_data = pt[res < 0.5]
    type_data = np.reshape(type_data, (type_data.shape[0] / 2, 2))
    plt.plot(type_data[:,0], type_data[:,1], 'o')

    # 第二类
    type_data = pt[res >= 0.5]
    type_data = np.reshape(type_data, (type_data.shape[0] / 2, 2))
    plt.plot(type_data[:,0], type_data[:,1], 'o')

    plt.show()

if __name__ == '__main__':
    #train_points()
    #train_symbols()
    test_symbols()