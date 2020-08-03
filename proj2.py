#!/usr/bin/env python
from os import sys
from collections import namedtuple
from enum import Enum
import math
import numpy as np

# A ProcessedImageData object is a pair of numpy arrays: images processed into features, and labels.
# It was more useful to organize data this way.
LabeledImage = namedtuple('LabeledImage', ['features', 'label'])
ProcessedImageData = namedtuple('ProcessedImageData', ['images', 'labels'])

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
    return f"data/facedata/facedata{mode.path_infix}{suffix}"

def read_image_data(mode, image_type):
    """Reads raw data from a data file."""
    with open(parse_path(mode, image_type, False), "r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
        number_of_images = len(lines) // image_type.rows
        return np.array([(lines[i*image_type.rows:(i+1)*image_type.rows]) for i in range(number_of_images)])

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

def read_labeled_images(mode, image_type):
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    feature_data = [extract_features(x) for x in raw_data]
    return [LabeledImage(features, label) for (features, label) in zip(feature_data, labels)]

def read_processed_images(mode, image_type):
    raw_data = read_image_data(mode, image_type)
    labels = read_label_data(mode, image_type)
    feature_data = [extract_features(x) for x in raw_data]
    return ProcessedImageData(feature_data, labels)

def calc_prior(value, data):
    #   count each value in the label_data and divide by number of labels to return the prior
    labels = np.array([label for (_, label) in data])
    count = np.count_nonzero(labels == value)
    print(count)
    print(len(labels))
    return count/len(labels)

def calc_feature_prob(feature, index, value, image_data):
    # for calculating the conditional probability of a certain pixel given a value
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

def calc_feature_probs(value, image_data, image_type):
    filtered_data = [labeled_image.features for labeled_image in image_data if labeled_image.label == value]
    total_occurrences = len(filtered_data)
    total_pixels = len(filtered_data[0])
    counts = np.zeros((3, total_pixels))
    for feature in range(3):
        for image_index in range(total_occurrences):
            for pixel_index in range(len(filtered_data[0])):
                if filtered_data[image_index][pixel_index] == feature:
                    counts[feature][pixel_index] += 1

    return np.array([count_sub / total_pixels for count_sub in counts])

def naive_bayes_one(image_type, i):
    label = None
    training_data = read_labeled_images(Mode.TRAINING, image_type)
    validation_data = read_labeled_images(Mode.VALIDATION, image_type)
    conditional_probabilities = [calc_feature_probs(val, training_data, ImageType.DIGIT) for val in range(image_type.categories)]
    priors = np.array([calc_prior(val, training_data) for val in range(image_type.categories)])
    features = validation_data[i].features
    max = -math.inf
    maxcat = -1
    for j in range(image_type.categories):
        prob = 0
        for index, f in enumerate(features):
            cond_prob = conditional_probabilities[j][f][index]
            if cond_prob != 0:
                prob += math.log(cond_prob)
        prior = priors[j]
        if prior != 0:
            prob += math.log(prior)
        print(f'Probability that image is {j}: {prob}')
        if prob > max:
            max = prob
            maxcat = j
    label = maxcat
    return label

def train_naive_bayes(image_type):
    print(f'Training classifier...')
    training_data = read_labeled_images(Mode.TRAINING, image_type)
    conditional_probabilities = [calc_feature_probs(val, training_data, ImageType.DIGIT) for val in range(image_type.categories)]
    priors = np.array([calc_prior(val, training_data) for val in range(image_type.categories)])
    print(f'Trained classifier for image type = {image_type.name}')
    return {'image_type': image_type, 'conditional_probabilities': conditional_probabilities, 'priors': priors}

def classify_naive_bayes(classifier_data, mode, indices):
    image_data = read_labeled_images(mode, classifier_data['image_type'])
    labels = []
    for i in indices:
        features = image_data[i].features
        max = -math.inf
        maxcat = None
        prob = 0
        for j in range(classifier_data['image_type'].categories):
            prob = 0
            for index, f in enumerate(features):
                cond_prob = classifier_data['conditional_probabilities'][j][f][index]
                if cond_prob != 0:
                    prob += math.log(cond_prob)
            prior = classifier_data['priors'][j]
            if prior != 0:
                prob += math.log(prior)
            if prob > max:
                max = prob
                maxcat = j
        labels.append((i, maxcat))
    for index, label in labels:
        print(f'Image {index} classified as: {label}')
    return labels

def check_correctness(classifier_out, mode, image_type):
    labels = read_label_data(mode, image_type)
    num_correct = 0
    total = len(classifier_out)
    for index, label in classifier_out:
        if labels[index] == label:
            num_correct += 1
    print(f'Got {num_correct} out of {total} correct: {(num_correct / total) * 100}%')

def naive_bayes(image_type):
    labels = []
    training_data = read_labeled_images(Mode.TRAINING, image_type)
    validation_data = read_labeled_images(Mode.VALIDATION, image_type)
    conditional_probabilities = [calc_feature_probs(val, training_data, ImageType.DIGIT) for val in range(image_type.categories)]
    priors = np.array([calc_prior(val, training_data) for val in range(image_type.categories)])
    for i in range(len(validation_data)):
        features = validation_data[i].features
        max = -math.inf
        maxcat = -1
        prob = 0
        for j in range(image_type.categories):
            for index, f in enumerate(features):
                cond_prob = conditional_probabilities[j][f][index]
                if cond_prob != 0:
                    prob += math.log(cond_prob)
            prior = priors[j]
            if prior != 0:
                prob += math.log(prior)
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
