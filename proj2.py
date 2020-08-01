#!/usr/bin/env python
from os import sys
from collections import namedtuple
from enum import Enum
import numpy as np
# A labeled image is a pair of features together with its label.
LabeledImage = namedtuple('LabeledImage', ['features', 'label'])

class ImageType(int, Enum):
    def __new__(cls, value, rows):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.rows = rows
        return obj

    DIGIT = (0, 28)
    FACE = (1, 70)

class Mode(str, Enum):
    def __new__(cls, value, path_infix):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.path_infix = path_infix
        return obj

    TRAINING = (0, "training")
    VALIDATION = (1, "validation")
    TEST = (2, "test")

def print_image(data, index):
    image = data[index]
    for line in image:
        print(line)

def print_labels(labels):
    print(labels)

def parse_path(mode, image_type, is_label):
    suffix = "labels" if is_label else ("images" if image_type == ImageType.DIGIT else "")
    if image_type == ImageType.DIGIT:
        return f"data/digitdata/{mode.path_infix}{suffix}"
    elif mode.TRAINING:
        return f"data/facedata/facedatatrain{suffix}"
    return f"data/facedata/facedata{mode.path_infix}{suffix}"

def read_image_data(mode, image_type):
    with open(parse_path(mode, image_type, False), "r", encoding="utf-8") as f:
        lines = f.readlines()
        number_of_images = len(lines) // image_type.rows
        return [lines[i*image_type.rows:(i+1)*image_type.rows] for i in range(number_of_images)]

def read_label_data(mode, image_type):
    with open(parse_path(mode, image_type, True), "r", encoding="utf-8") as f:
        return [int(x) for x in f]

def extract_features(raw_data):
    features = []
    for line in raw_data:
        for c in line:
            if c == ' ':
                features.append(0)
            elif c == '+':
                features.append(1)
            else:
                features.append(2)
    return features

def read_labeled_images(mode, image_type):
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    feature_data = extract_features(raw_data)
    return [LabeledImage(features, label) for (features, label) in zip(feature_data, labels)]

def calcPrior(value, label_data):
    #   count each value in the label_data and divide by number of labels to return the prior
    labels = np.asarray(label_data)
    count = np.count_nonzero(labels == value)
    print(count)
    print(len(labels))
    return count/len(labels)

def calcFeatureProb(feature, index, value, data, label_data):
    # for calculating the conditional probability of a certain pixel given a value
    i = 0
    count = 0
    for l in label_data:
        if l == value:
            image = data[i]
            if extract_features(image)[index] == feature:
                count += 1
        i += 1
    return(count/np.count_nonzero(np.asarray(label_data) == value))

def NaiveBayes(type):
    labels = []
    if(type == "DIGIT"):
        traindata = read_image_data(Mode.TRAINING, ImageType.DIGIT)
        trainlabels = read_label_data(Mode.TRAINING, ImageType.DIGIT)
        validationdata = read_image_data(Mode.VALIDATION, ImageType.DIGIT)
        validationlabels = read_label_data(Mode.VALIDATION, ImageType.DIGIT)
        cats = 10
    else:
        traindata = read_image_data(Mode.TRAINING, ImageType.FACE)
        trainlabels = read_label_data(Mode.TRAINING, ImageType.FACE)
        validationdata = read_image_data(Mode.VALIDATION, ImageType.FACE)
        validationlabels = read_label_data(Mode.VALIDATION, ImageType.FACE)
        cats = 2
    for i in range(cats):
        # do stuff. Still need to code this

if __name__ == '__main__':
    #print(calcPrior(5, read_label_data(Mode.TRAINING, ImageType.DIGIT)))
    print(calcFeatureProb(0,400,9,read_image_data(Mode.TRAINING, ImageType.DIGIT),read_label_data(Mode.TRAINING, ImageType.DIGIT)))




