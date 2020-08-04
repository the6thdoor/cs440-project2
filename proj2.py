#!/usr/bin/env python
"""Implementation of the Naive Bayes classifier."""
from collections import namedtuple
from enum import Enum
import argparse
import numpy as np

LabeledImage = namedtuple('LabeledImage', ['features', 'label', 'index'])

class ProcessedImageData:
    """A representation of the image data as a pair of arrays: features and labels.
    Also provides access to image data in LabeledImage form, to be accessed by index."""
    def __init__(self, features, labels, indices):
        self.features = features
        self.labels = labels
        self.indices = indices

    def __iter__(self):
        size = min(len(self.features), len(self.labels))
        for i in range(size):
            yield LabeledImage(self.features[i], self.labels[i], self.indices[i])

    def __getitem__(self, i):
        return LabeledImage(self.features[i], self.labels[i], self.indices[i])

    def __setitem__(self, i, tup):
        self.features[i] = tup[0]
        self.labels[i] = tup[1]
        self.indices[i] = tup[2]

    def sample_percent(self, percentage):
        count = int(len(self.features) * (percentage / 100))
        indices = np.random.randint(0, high=len(self.features), size=count)
        return ProcessedImageData(self.features[indices], self.labels[indices], indices)

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
    return ProcessedImageData(features, labels, np.arange(len(features)))

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

def train_naive_bayes_partial(image_type, smoothing, percentage):
    """Uses a partial set of the training data to train the Naive Bayes classifier."""
    print('Loading training data...')
    training_data = read_processed_images(Mode.TRAINING, image_type)
    print(f'Selecting {percentage}% of the training data at random...')
    partial_data = training_data.sample_percent(percentage)
    print('Training classifier...')
    conditionals = calc_feature_probs(image_type.categories, partial_data, smoothing)
    priors = calc_priors(image_type.categories, partial_data)
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
    # for index, label in labels:
    #     print(f'Image {index} classified as: {label}')
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

def old_main(argv):
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

def run_classifier_bayes(mode, image_type, indices, smoothing, percentage):
    dat = train_naive_bayes(image_type, smoothing) if percentage == 100 else train_naive_bayes_partial(image_type, smoothing, percentage)
    output = classify_naive_bayes(dat, mode, indices)
    check_correctness(output, mode, image_type)

def run_classifier_perceptron(mode, image_type, indices, smoothing, percentage):
    pass

def main():
    parser = argparse.ArgumentParser(description='Implementation of the Naive Bayes and Perceptron classifiers')
    parser.add_argument('--classifier', metavar='C', help='classifier to use', choices=['BAYES', 'PERCEPTRON'], required=True)
    parser.add_argument('--mode', help='image class to test', choices=['VALIDATION', 'TEST'], default='TEST')
    parser.add_argument('--type', help='image type to train', choices=['DIGIT', 'FACE'], required=True)
    parser.add_argument('--range', metavar=('START', 'END_EXCLUSIVE'), nargs=2, type=int, help='Range of data to test', required=True)
    parser.add_argument('--trainpercent', metavar='PERCENT', type=int, help='the percent of training data to use (int out of 100)', default=100)
    parser.add_argument('--smoothing', metavar='K', type=int, help='Laplace smoothing constant', default=2)
    parser.add_argument('--debug', help='skips execution for debugging purposes', action='store_true')
    args = parser.parse_args()
    image_type = ImageType.DIGIT if args.type == 'DIGIT' else ImageType.FACE
    mode = Mode.TEST if args.mode == 'TEST' else Mode.VALIDATION
    run = run_classifier_bayes if args.classifier == 'BAYES' else run_classifier_perceptron
    if not args.debug:
        run(mode, image_type, range(args.range[0], args.range[1]), args.smoothing, args.trainpercent)
    else:
        print('Debug mode: Welcome.')

if __name__ == '__main__':
    main()
