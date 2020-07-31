#!/usr/bin/env python
"""Reads data from project files in order to:
    1) extract features from the raw data
    2) attach the corresponding label for the classifier to process.

    It may be better to let this file be responsible for just reading data from files.
    The classifiers (Naive Bayes and Perceptron) should go in their own files.
"""
from os import sys
from collections import namedtuple
from enum import Enum

# A labeled image is a pair of features together with its label.
LabeledImage = namedtuple('LabeledImage', ['features', 'label'])

class ImageType(int, Enum):
    """A representation of the image type, either digit or face.

    Attributes
    ----------
    rows : int
        the number of rows in the corresponding image type
    """
    def __new__(cls, value, rows):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.rows = rows
        return obj

    DIGIT = (0, 28)
    FACE = (1, 70)

class Mode(str, Enum):
    """A representation of the mode of the image, either training, validation, or test.

    Attributes
    ----------
    path_infix : str
        the string in the mode's file path which corresponds to the image mode
    """
    def __new__(cls, value, path_infix):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.path_infix = path_infix
        return obj

    TRAINING = (0, "training")
    VALIDATION = (1, "validation")
    TEST = (2, "test")

def print_image(data, index):
    """Prints a raw image (array of strings) to stdout."""
    image = data[index]
    for line in image:
        print(line)

def parse_path(mode, image_type, is_label):
    """Returns the file path to the corresponding image type and mode."""
    suffix = "labels" if is_label else ("images" if image_type == ImageType.DIGIT else "")
    if image_type == ImageType.DIGIT:
        return f"data/digitdata/{mode.path_infix}{suffix}"
    return f"data/facedata/facedata{mode.path_infix}{suffix}"

def read_image_data(mode, image_type):
    """Reads raw data from a data file."""
    with open(parse_path(mode, image_type, False), "r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
        number_of_images = len(lines) // image_type.rows
        return [lines[i*image_type.rows:(i+1)*image_type.rows] for i in range(number_of_images)]

def read_label_data(mode, image_type):
    """Reads labels from a label file."""
    with open(parse_path(mode, image_type, True), "r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
        return map(int, lines)

def extract_features(raw_data):
    """Parses raw text data into features, represented by an int from 0-2 for the pixel brightness.
    0 is an empty pixel, 1 is a half-full pixel (+), and 2 is a full pixel (#).
    """
    features = []
    for line in raw_data:
        for char in line:
            if char == ' ':
                features.append(0)
            elif char == '+':
                features.append(1)
            else:
                features.append(2)
    return features

def read_labeled_images(mode, image_type):
    """Reads data and label files and compiles them into an array of labeled images."""
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    feature_data = extract_features(raw_data)
    return [LabeledImage(features, label) for (features, label) in zip(feature_data, labels)]

if __name__ == '__main__':
    print_image(read_image_data(Mode.TRAINING, ImageType.DIGIT), int(sys.argv[1]))
