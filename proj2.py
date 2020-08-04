#!/usr/bin/env python
"""Implementation of the Naive Bayes classifier."""
from collections import namedtuple
from enum import Enum
import numpy as np

LabeledImage = namedtuple('LabeledImage', ['features', 'label'])

class ProcessedImageData:
    """A representation of the image data as a pair of arrays: features and labels.
    Also provides access to image data in LabeledImage form, to be accessed by index."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __iter__(self):
        size = min(len(self.features), len(self.labels))
        for i in range(size):
            yield LabeledImage(self.features[i], self.labels[i])

    def __getitem__(self, i):
        return LabeledImage(self.features[i], self.labels[i])

    def __setitem__(self, i, tup):
        self.features[i] = tup[0]
        self.labels[i] = tup[1]

class ImageType(int, Enum):
    """A representation of the image type, either digit or face.

    Attributes
    ----------
    rows : int
        the number of rows in the corresponding image type
    categories : int
        the number of possible labels
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
    if mode == Mode.TRAINING:
        return f"data/facedata/facedatatrain{suffix}"
    return f"data/facedata/facedata{mode.path_infix}{suffix}"

def read_image_data(mode, image_type):
    """Reads raw data from a data file."""
    txtdata = np.loadtxt(parse_path(mode, image_type, False), dtype=str, delimiter='\n', comments=None)
    num_rows = image_type.rows
    num_images = len(txtdata) // num_rows
    return txtdata.reshape((num_images, num_rows))

def read_label_data(mode, image_type):
    """Reads labels from a label file."""
    return np.loadtxt(parse_path(mode, image_type, True), dtype=int, delimiter='\n')

def extract_features_debug(raw_data):
    """Parses raw text data into features, represented by an int from 0-2 for the pixel brightness.
    0 is an empty pixel, 1 is a half-full pixel (+), and 2 is a full pixel (#).
    """
    width = len(raw_data[0])
    num_features = len(raw_data) * width
    features = np.zeros((num_features, 3), dtype=int)
    for row, line in enumerate(raw_data):
        for col, char in enumerate(line):
            if char == ' ':
                features[col + row * width] = 0
            elif char == '+':
                features[col + row * width] = 1
            elif char == '#':
                features[col + row * width] = 2
    return features


def extract_features(raw_data):
    """Parses raw text data into features, represented by an int from 0-2 for the pixel brightness.
    0 is an empty pixel, 1 is a half-full pixel (+), and 2 is a full pixel (#).
    """
    width = len(raw_data[0])
    num_features = len(raw_data) * width
    features = np.zeros((num_features, 3), dtype=bool)
    for row, line in enumerate(raw_data):
        for col, char in enumerate(line):
            if char == ' ':
                features[col + row * width][0] = True
            elif char == '+':
                features[col + row * width][1] = True
            elif char == '#':
                features[col + row * width][2] = True
    return features

def read_processed_images(mode, image_type):
    """Reads data and label files and compiles them into an array of labeled images."""
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    features = np.apply_along_axis(extract_features, 1, raw_data)
    return ProcessedImageData(features, labels)

def calc_priors(categories, data):
    """Calculates the prior probabilities of each label by counting the occurrences of the label
       and dividing by the total number of labels."""
    counts = np.zeros(categories)
    for val in range(categories):
        counts[val] = np.count_nonzero(data.labels == val)
    return counts / len(data.labels)

def calc_feature_probs(categories, image_data, smoothing):
    """Computes the conditional probability for each feature-pixel pair, given a certain label."""
    counts = np.array([np.sum(image_data.features[image_data.labels == value], axis=0) + smoothing for value in range(categories)])
    denoms = np.array([np.count_nonzero(image_data.labels == value) + (smoothing * 3) for value in range(categories)])
    return counts / denoms[:, np.newaxis, np.newaxis]

def train_naive_bayes(image_type, smoothing):
    """Uses the training data of the image type to assemble Bayesian probability data."""
    print('Loading training data...')
    training_data = read_processed_images(Mode.TRAINING, image_type)
    print('Training classifier...')
    conditionals = calc_feature_probs(image_type.categories, training_data, smoothing)
    priors = calc_priors(image_type.categories, training_data)
    print(f'Trained classifier for image type = {image_type.name}')
    return {'image_type': image_type, 'conditionals': conditionals, 'priors': priors}

def classify_naive_bayes(classifier_data, mode, indices):
    """Classifies the given images using the given probability data."""
    image_data = read_processed_images(mode, classifier_data['image_type'])
    labels = []
    log_priors = np.log(classifier_data['priors'])
    log_conditionals = np.log(classifier_data['conditionals'])
    categories = classifier_data['image_type'].categories
    for i in indices:
        probabilities = np.zeros(categories)
        for cur_label in range(categories):
            log_prior = log_priors[cur_label]
            log_conds = np.sum(log_conditionals[cur_label, image_data[i].features])
            probabilities[cur_label] = log_prior + log_conds
        labels.append((i, np.argmax(probabilities)))
    for index, label in labels:
        print(f'Image {index} classified as: {label}')
    return labels

def check_correctness(classifier_out, mode, image_type):
    """Checks how many images were correctly classified."""
    labels = read_label_data(mode, image_type)
    num_correct = 0
    total = len(classifier_out)
    for index, label in classifier_out:
        if labels[index] == label:
            num_correct += 1
    print(f'Got {num_correct} out of {total} correct: {(num_correct / total) * 100}%')

if __name__ == '__main__':
    DEBUG_MODE = input('Skip execution for interactive debugging? [y/n] ')
    if DEBUG_MODE in ('y', 'Y'):
        print('Welcome.')
    else:
        SELECT_IMAGE_TYPE = 'y'
        while SELECT_IMAGE_TYPE in ('y', 'Y'):
            img_type_str = input('Select an image type, either [d]igit or [f]ace: ')
            while img_type_str not in ('d', 'f', 'D', 'F'):
                img_type_str = input('Invalid image type. Please select either [d]igit or [f]ace: ')
            img_type = ImageType.DIGIT if img_type_str == 'd' else ImageType.FACE
            SELECT_NEW_SMOOTHING = 'y'
            while SELECT_NEW_SMOOTHING in ('y', 'Y'):
                k = int(input('Select an integer to use as a smoothing parameter: '))
                dat = train_naive_bayes(img_type, k)
                SELECT_IMAGE_MODE = 'y'
                while SELECT_IMAGE_MODE in ('y', 'Y'):
                    check_img_mode_str = input('Would you like to use [v]alidation or [t]est data? ')
                    while check_img_mode_str not in ('v', 't', 'V', 'T'):
                        check_img_mode_str = input('Invalid mode. Select [v]alidation or [t]est: ')
                    TEST_NEW_IMAGES = 'y'
                    while TEST_NEW_IMAGES in ('y', 'Y'):
                        check_img_mode = Mode.VALIDATION if check_img_mode_str == 'v' else Mode.TEST
                        images_str = input('Enter a range of indices to test, e.g. 2,4: ').split(',')
                        images = range(int(images_str[0]), int(images_str[1]))
                        output = classify_naive_bayes(dat, check_img_mode, images)
                        check_correctness(output, check_img_mode, img_type)
                        TEST_NEW_IMAGES = input('Would you like to test a new range? [y/n] ')
                    SELECT_IMAGE_MODE = input('Would you like to test a different mode? [y/n] ')
                SELECT_NEW_SMOOTHING = input('Would you like to test a different smoothing value? [y/n] ')
            SELECT_IMAGE_TYPE = input('Would you like to test a different image type? [y/n] ')
        print('Done. Exiting...')
