#!/usr/bin/env python
"""Implementation of the Naive Bayes classifier."""
from collections import namedtuple
from enum import Enum
import math
import numpy as np

# A ProcessedImageData object is a pair of numpy arrays: images processed into features, and labels.
# It was more useful to organize data this way.
LabeledImage = namedtuple('LabeledImage', ['features', 'label'])

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
    with open(parse_path(mode, image_type, False), "r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
        num_rows = image_type.rows
        num_images = len(lines) // num_rows
        return np.array([(lines[i*num_rows:(i+1)*num_rows]) for i in range(num_images)])

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
    return np.array(features)

def read_labeled_images(mode, image_type):
    """Reads data and label files and compiles them into an array of labeled images."""
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    feature_data = [extract_features(x) for x in raw_data]
    return [LabeledImage(features, label) for (features, label) in zip(feature_data, labels)]

def calc_prior(value, data):
    """Counts each value in the label data and divides by number of labels to return the prior."""
    labels = np.array([label for (_, label) in data])
    count = np.count_nonzero(labels == value)
    print(count)
    print(len(labels))
    return count/len(labels)

def calc_feature_probs(value, image_data, smoothing):
    """Computes the conditional probability for each feature-pixel pair, given a certain label."""
    value_images = [image.features for image in image_data if image.label == value]
    total_occurrences = len(value_images)
    total_pixels = len(value_images[0])
    counts = np.full((3, total_pixels), smoothing)
    for feature in range(3):
        for image_index in range(total_occurrences):
            for pixel_index in range(len(value_images[0])):
                if value_images[image_index][pixel_index] == feature:
                    counts[feature][pixel_index] += 1
    denominator = total_occurrences + (smoothing * 3)
    return np.array([count_sub / denominator for count_sub in counts])

def train_naive_bayes(image_type, smoothing):
    """Uses the training data of the image type to assemble Bayesian probability data."""
    print('Training classifier...')
    training_data = read_labeled_images(Mode.TRAINING, image_type)
    conditionals = [calc_feature_probs(val, training_data, smoothing) for val in range(image_type.categories)]
    priors = np.array([calc_prior(val, training_data) for val in range(image_type.categories)])
    print(f'Trained classifier for image type = {image_type.name}')
    return {'image_type': image_type, 'conditionals': conditionals, 'priors': priors}

def classify_naive_bayes(classifier_data, mode, indices):
    """Classifies the given images using the given probability data."""
    image_data = read_labeled_images(mode, classifier_data['image_type'])
    labels = []
    for i in indices:
        prob_max = -math.inf
        maxcat = None
        for current_label in range(classifier_data['image_type'].categories):
            prob = 0
            for index, feature in enumerate(image_data[i].features):
                cond_prob = classifier_data['conditionals'][current_label][feature][index]
                if cond_prob != 0:
                    prob += math.log(cond_prob)
            prior = classifier_data['priors'][current_label]
            if prior != 0:
                prob += math.log(prior)
            if prob > prob_max:
                prob_max = prob
                maxcat = current_label
        labels.append((i, maxcat))
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
