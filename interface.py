#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: interface.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 15:28
# --------------------------------------------------------
import numpy as np
import cv2
from sklearn.svm import LinearSVC
import time
import os


class TemplateMatching(object):
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    def __init__(self):
        self.height = None
        self.matchingRMin = 15.0
        self.height_var = 8
        self.symbol_ratio = 2.0 / 7
        self.func_params = [4.3465, 0]
        self.method = cv2.TM_CCOEFF_NORMED
        self.isDetectedCircle = False
        self.scaleMatching = None
        self.template = None
        self.rect = None
        self.scoreThresh = 0.6

    def getScaleRatioFromHeight(self, params, h, hmin=10, hmax=120):
        if h <= hmin: h = hmin
        if h >= hmax: h = hmax
        a, b = params
        return a / h + b

    def createTemplate(self, ratio=2 / 7, rmax=960):
        template = np.zeros(shape=(rmax * 2, rmax * 2, 3), dtype=np.uint8)
        center = (rmax, rmax)
        cv2.circle(template, center, int(rmax), (255, 0, 0), -1, 8)
        cv2.circle(template, center, rmax * 2 / 7, (0, 0, 255), -1, 8)
        return template

    def templateMatchingInit(self, height, im, matchingRMin=15.0, height_var=8):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im_gray
        fin_score = None
        fin_rect = None
        fin_template = None
        fin_scaleMatching = None
        for i in range(height - height_var, height + height_var):
            scaleRatio = self.getScaleRatioFromHeight(h=i, params=self.func_params)
            r = int(scaleRatio * im.shape[1] / 2)
            template = self.createTemplate(rmax=r)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = template_gray

            # scale the template image
            if r > matchingRMin:
                scaleMatching = matchingRMin / r
                template = cv2.resize(template, None, None, scaleMatching, scaleMatching, cv2.INTER_CUBIC)
                im_matching = cv2.resize(im, None, None, scaleMatching, scaleMatching, cv2.INTER_CUBIC)

            result = cv2.matchTemplate(im_matching, template, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            w, h = template.shape[::-1]
            if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                if fin_score is None or fin_score > min_val:
                    fin_score = min_val
                    fin_rect = [int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])]
                    fin_template = template
                    fin_scaleMatching = scaleMatching
            else:
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                if fin_score is None or fin_score < max_val:
                    fin_score = max_val
                    fin_rect = [int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])]
                    fin_template = template
                    fin_scaleMatching = scaleMatching
        rect = None
        if fin_score >= self.scoreThresh:
            self.isDetectedCircle = True
            self.scaleMatching = fin_scaleMatching
            self.template = fin_template
            rect = [int(fin_rect[0] / fin_scaleMatching + 0.5), int(fin_rect[1] / fin_scaleMatching + 0.5),
                    int(fin_rect[2] / fin_scaleMatching + 0.5), int(fin_rect[3] / fin_scaleMatching + 0.5)]
        else:
            self.isDetectedCircle = False
        return rect

    def templateMatching(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im_gray
        im_matching = cv2.resize(im, None, None, self.scaleMatching, self.scaleMatching, cv2.INTER_CUBIC)
        result = cv2.matchTemplate(im_matching, self.template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            score = min_val
        else:
            top_left = max_loc
            score = max_val
        bottom_right = (top_left[0] + self.template.shape[1], top_left[1] + self.template.shape[0])

        rect = [int(top_left[0] / self.scaleMatching + 0.5), int(top_left[1] / self.scaleMatching + 0.5),
                int(bottom_right[0] / self.scaleMatching + 0.5), int(bottom_right[1] / self.scaleMatching + 0.5)]
        if score < self.scoreThresh:
            self.isDetectedCircle = False
        return rect

    def getRecogintionPatch(self, rect, im):
        center_x = (rect[0] + rect[2]) / 2
        center_y = (rect[1] + rect[3]) / 2
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        target_x_min = int(center_x - w / 7 * 3 / 2)
        target_y_min = int(center_y - h / 7 * 3 / 2)
        target_x_max = int(center_x + w / 7 * 3 / 2)
        target_y_max = int(center_y + h / 7 * 3 / 2)
        return im[target_y_min:target_y_max, target_x_min:target_x_max]


class SymbolRecognition(object):
    def __init__(self):
        # init the cv2 hog class
        self.winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(self.winSize, blockSize, blockStride, cellSize, nbins)
        # init the sklearn linearSVC
        modelpath = 'model.npz'
        self.model = LinearSVC()
        self.load_model(modelpath)

    def load_model(self, path):
        r = np.load(path)
        classes_ = r['classes']
        coef_ = r['coef']
        intercept_ = r['intercept']
        self.model.__setattr__('classes_', classes_)
        self.model.__setattr__('coef_', coef_)
        self.model.__setattr__('intercept_', intercept_)

    def extract_hog(self, im):
        descriptor = self.hog.compute(im)
        if descriptor is None:
            return None
        return descriptor.ravel()

    def predict(self, im):
        im = cv2.resize(im, self.winSize, interpolation=cv2.INTER_CUBIC)
        feature = self.extract_hog(im)
        return self.model.predict(feature.reshape(1, -1))[0]


def singleImageTMSR():
    imgs = sorted([filename for filename in os.listdir('demo') if filename[-4:] == '.JPG'])
    init_height = 100
    TM = TemplateMatching()
    SR = SymbolRecognition()
    for i in range(len(imgs)):
        impath = 'demo/{}'.format(imgs[i])
        print impath
        im = cv2.imread(impath)
        height = init_height - i * 5
        time_start = time.time()
        time_start_m = time.time()
        score, rect = TM.templateMatchingInit(height=height, im=im)
        recoPatch = TM.getRecogintionPatch(rect, im)
        time_end_m = time.time()
        print 'matching time cost: {}'.format(time_end_m - time_start_m)
        time_start_p = time.time()
        decision = SR.predict(recoPatch)
        time_end_p = time.time()
        print 'predicting time cost: {}'.format(time_end_p - time_start_p)
        if decision == 0:
            text = 'F'
        else:
            text = 'T'
        time_end = time.time()
        print 'matching and predicting time cost: {}'.format(time_end - time_start)
        # for show
        im = cv2.resize(im, None, None, 0.25, 0.25, cv2.INTER_CUBIC)
        cv2.imshow('rectPatch', recoPatch)
        cv2.putText(im, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
        cv2.imshow('predict', im)
        cv2.waitKey(1)


def videoTMSR():
    TM = TemplateMatching()
    SR = SymbolRecognition()
    videopath = 'demo/DJI_0056.mp4'
    videoCapture = cv2.VideoCapture(videopath)
    height = 30
    success, frame = videoCapture.read()
    while success:
        if TM.isDetectedCircle is False:
            rect = TM.templateMatchingInit(height=height, im=frame)
        else:
            rect = TM.templateMatchingInit(height=height, im=frame)
        if TM.isDetectedCircle:
            recoPatch = TM.getRecogintionPatch(rect, frame)
            decision = SR.predict(recoPatch)
        else:
            decision = -1

        if decision == -1:
            text = 'No circle matching'
        elif decision == 0:
            text = 'F'
        else:
            text = 'T'

        frame = cv2.resize(frame, None, None, 0.25, 0.25, cv2.INTER_CUBIC)
        if TM.isDetectedCircle:
            cv2.imshow('rectPatch', recoPatch)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('predict', frame)
        cv2.waitKey(1)
        success, frame = videoCapture.read()


if __name__ == "__main__":
    # singleImageTMSR()
    videoTMSR()
