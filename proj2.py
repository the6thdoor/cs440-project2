#!/usr/bin/env python
from os import sys
from collections import namedtuple
from enum import Enum

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

def parse_path(mode, image_type, is_label):
    suffix = "labels" if is_label else ("images" if image_type == ImageType.DIGIT else "")
    if image_type == ImageType.DIGIT:
        return f"data/digitdata/{mode.path_infix}{suffix}"
    return f"data/facedata/facedata{mode.path_infix}{suffix}"

def read_image_data(mode, image_type):
    with open(parse_path(mode, image_type, False), "r", encoding="utf-8") as f:
        lines = f.readlines()
        number_of_images = len(lines) // image_type.rows
        return [lines[i*image_type.rows:(i+1)*image_type.rows] for i in range(number_of_images)]

def read_label_data(mode, image_type):
    with open(parse_path(mode, image_type, True), "r", encoding="utf-8") as f:
        lines = f.readlines()
        return map(int, lines)

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

if __name__ == '__main__':
    print_image(read_image_data(Mode.TRAINING, ImageType.DIGIT), int(sys.argv[1]))
