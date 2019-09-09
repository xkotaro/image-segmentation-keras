#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras_segmentation
import os

model = keras_segmentation.pretrained.pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset
# model = keras_segmentation.pretrained.pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset
# model = keras_segmentation.pretrained.pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset


path = ''
list = os.listdir(path)

for i in range(len(list2)):
    inp = list[i]
    out_fname = os.path.splitext(list[i])[0]  + '_seg' + '.png'
    out = model.predict_segmentation(
    inp=inp,
    out_fname=out_fname
)


