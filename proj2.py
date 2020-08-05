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
        """Returns a random sample of a given percentage of the image data."""
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

def train_naive_bayes(image_type, smoothing, percentage):
    """Uses the training data of the image type to assemble Bayesian probability data."""
    print('Loading training data...')
    training_data = read_processed_images(Mode.TRAINING, image_type)
    if percentage != 100:
        print(f'Selecting {percentage}% of the training data at random...')
        training_data = training_data.sample_percent(percentage)
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

def train_perceptron(image_type, iterations, percentage):
    """Learns the proper weights for the Perceptron classifier
       over a given number of iterations."""
    print('Loading training data...')
    image_data = read_processed_images(Mode.TRAINING, image_type)
    if percentage != 100:
        print(f'Selecting {percentage}% of the training data at random...')
        image_data = image_data.sample_percent(percentage)
    print(f'Training classifier...')
    num_labels = len(image_data.labels)
    num_pixels = len(image_data.features[0])
    weights = np.random.rand(num_labels, num_pixels)
    encoding = np.repeat(np.arange(3).reshape(1, 3), num_pixels, axis=0)
    for i in range(iterations):
        for image in image_data:
            encoded_features = encoding[image.features]
            scores = np.array([np.dot(encoded_features, weights[cat]) for cat in range(image_type.categories)])
            guess = np.argmax(scores)
            if guess != image.label:
                weights[image.label] += encoded_features
                weights[guess] -= encoded_features
        print(f'Completed iteration {i}.')
    print(f'Trained Perceptron classifier for image type = {image_type.name}')
    return {'image_type': image_type, 'weights': weights}

def classify_perceptron(classifier_data, mode, indices):
    """Uses the weights learned according to the Perceptron algorithm
       to classify a validation/test image."""
    image_type = classifier_data['image_type']
    image_data = read_processed_images(mode, image_type)
    num_pixels = len(image_data.features[0])
    weights = classifier_data['weights']
    encoding = np.repeat(np.arange(3).reshape(1, 3), num_pixels, axis=0)
    labels = []
    for i in indices:
        image = image_data[i]
        encoded_features = encoding[image.features]
        scores = np.array([np.dot(encoded_features, weights[cat]) for cat in range(image_type.categories)])
        guess = np.argmax(scores)
        labels.append((i, guess)) # :)
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

def run_classifier_bayes_smoothing(mode, image_type, indices, percentage, smoothing):
    """Runs the Naive Bayes classifier from start to finish
       using a fixed, predetermined smoothing constant."""
    smoothing = input('Select a smoothing value [default=2]: ')
    dat = train_naive_bayes(image_type, smoothing, percentage)
    output = classify_naive_bayes(dat, mode, indices)
    check_correctness(output, mode, image_type)

def run_classifier_bayes(mode, image_type, indices, percentage):
    """Runs the Naive Bayes classifier from start to finish,
       obtaining the smoothing constant from stdin."""
    smoothing = input('Select a smoothing value [default=2]: ')
    try:
        smoothing = int(smoothing)
    except ValueError:
        print('Not a valid input. Using default smoothing value of 2.')
        smoothing = 2
    run_classifier_bayes_smoothing(mode, image_type, indices, percentage, smoothing)

def run_classifier_perceptron_iterations(mode, image_type, indices, percentage, iterations):
    """Runs the Perceptron classifier from start to finish
       over a fixed, predetermined number of iterations."""
    dat = train_perceptron(image_type, iterations, percentage)
    output = classify_perceptron(dat, mode, indices)
    check_correctness(output, mode, image_type)

def run_classifier_perceptron(mode, image_type, indices, percentage):
    """Runs the Perceptron classifier from start to finish,
       obtaining the number of iterations from stdin."""
    iterations = input('Select number of iterations [default=5]: ')
    try:
        iterations = int(iterations)
    except ValueError:
        print('Not a valid input. Using default number of iterations (5).')
        iterations = 5
    run_classifier_perceptron_iterations(mode, image_type, indices, percentage, iterations)

def main():
    """Command line interface for the Naive Bayes and Perceptron classifiers."""
    parser = argparse.ArgumentParser(description='Implementation of the Naive Bayes and Perceptron classifiers')
    parser.add_argument('--classifier', help='classifier to use', choices=['BAYES', 'PERCEPTRON'], required=True)
    parser.add_argument('--mode', help='image class to test', choices=['VALIDATION', 'TEST'], default='TEST')
    parser.add_argument('--type', help='image type to train', choices=['DIGIT', 'FACE'], required=True)
    parser.add_argument('--range', metavar=('START', 'END_EXCLUSIVE'), nargs=2, type=int, help='Range of data to test', required=True)
    parser.add_argument('--trainpercent', metavar='PERCENT', type=int, help='the percent of training data to use (int out of 100)', default=100)
    parser.add_argument('--debug', help='skips execution for debugging purposes', action='store_true')
    args = parser.parse_args()
    image_type = ImageType.DIGIT if args.type == 'DIGIT' else ImageType.FACE
    mode = Mode.TEST if args.mode == 'TEST' else Mode.VALIDATION
    run = run_classifier_bayes if args.classifier == 'BAYES' else run_classifier_perceptron
    if not args.debug:
        run(mode, image_type, range(args.range[0], args.range[1]), args.trainpercent)
    else:
        print('Debug mode: Welcome.')

if __name__ == '__main__':
    main()
