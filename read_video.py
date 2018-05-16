#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: read_video.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 16:11
# --------------------------------------------------------

import cv2
import os

video_path = 'F:/uva/Mavic-20180509-天津海教园/DJI_0053.MP4'
save_path = 'records/50m_rotation_T'
if not os.path.exists(save_path):
    os.mkdir(save_path)

if __name__ == '__main__':
    videoCapture = cv2.VideoCapture(video_path)
    frameid = 0
    success, frame = videoCapture.read()
    while success and frameid < 300:
        imgpath = os.path.join(save_path, '{:0>6}.jpg'.format(frameid))
        cv2.imwrite(imgpath, frame)
        frameid += 1
        print imgpath
        success, frame = videoCapture.read()


