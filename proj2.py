#!/usr/bin/env python
from os import sys
from collections import namedtuple
from enum import Enum
import math
import numpy as np

# A ProcessedImageData object is a pair of numpy arrays: images processed into features, and labels.
# It was more useful to organize data this way.
ProcessedImageData = namedtuple('ProcessedImageData', ['images', 'labels'])

class ImageType(int, Enum):
    """A representation of the image type, either digit or face.

    Attributes
    ----------
    rows : int
        the number of rows in the corresponding image type
    """
    def __new__(cls, value, rows, categories):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.rows = rows
        obj.categories = categories
        return obj

    DIGIT = (0, 28, 10)
    FACE = (1, 70, 2)

class Mode(int, Enum):
    """A representation of the mode of the image, either training, validation, or test.

    Attributes
    ----------
    path_infix : str
        the string in the mode's file path which corresponds to the image mode
    """
    def __new__(cls, value, path_infix):
        obj = int.__new__(cls, value)
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
        return np.array([np.array(lines[i*image_type.rows:(i+1)*image_type.rows]) for i in range(number_of_images)])

def read_label_data(mode, image_type):
    """Reads labels from a label file."""
    with open(parse_path(mode, image_type, True), "r", encoding="utf-8") as datafile:
        return np.array([int(x) for x in datafile])

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
            elif char == '#':
                features.append(2)
    return features

def read_processed_images(mode, image_type):
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    feature_data = [extract_features(x) for x in raw_data]
    return ProcessedImageData(feature_data, labels)

def calc_prior(value, labels):
    #   count each value in the label_data and divide by number of labels to return the prior
    count = np.count_nonzero(labels == value)
    print(count)
    print(len(labels))
    return count/len(labels)

def calc_feature_prob(feature, index, value, image_data):
    # for calculating the conditional probability of a certain pixel given a value)
    i = 0
    count = 0
    labels = image_data.labels
    for l in labels:
        if l == value:
            image_feature = image_data.images[i][index]
            if image_feature == feature:
                count += 1
        i += 1
    return(count/np.count_nonzero(labels == value))

def naive_bayes(image_type):
    labels = []
    training_data = read_processed_images(Mode.TRAINING, image_type)
    validation_data = read_processed_images(Mode.VALIDATION, image_type)
    for i in range(len(validation_data)):
        features = validation_data.images[i]
        max = -math.inf
        maxcat = -1
        prob = 0
        for j in range(image_type.categories):
            index = 0
            for f in features:
                temp = calc_feature_prob(f, index, j, training_data)
                if temp != 0:
                    prob += math.log(temp)
                index += 1
            temp = calc_prior(j, training_data.labels)
            if temp != 0:
                print(temp)
                prob += math.log(temp)
            if prob > max:
                max = prob
                maxcat = j
        labels.append(maxcat)
    return labels

if __name__ == '__main__':
    #print(calc_prior(5, read_label_data(Mode.TRAINING, ImageType.DIGIT)))
    if sys.argv[1] == 'debug':
        print('hi!')
    else:
        print(calc_feature_prob(0, 400, 9, read_processed_images(Mode.TRAINING, ImageType.DIGIT)))
        print(naive_bayes(ImageType.DIGIT))
