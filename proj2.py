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

class ImageType(str, Enum):
    """A representation of the image type, either digit or face.

    Attributes
    ----------
    rows : int
        the number of rows in the corresponding image type
    categories : int
        the number of possible labels
    """
    def __new__(cls, value, rows, categories):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.rows = rows
        obj.categories = categories
        obj._image_data = None
        return obj

    DIGIT = ("DIGIT", 28, 10)
    FACE = ("FACE", 70, 2)

    @property
    def image_data(self):
        if self._image_data == None:
            self._image_data = process_all_image_data(self)
        return self._image_data

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

    TRAINING = ("TRAINING", "training")
    VALIDATION = ("VALIDATION", "validation")
    TEST = ("TEST", "test")

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

def process_all_image_data(image_type):
    print('Preloading all image data...')
    training_data = read_processed_images(Mode.TRAINING, image_type)
    validation_data = read_processed_images(Mode.VALIDATION, image_type)
    test_data = read_processed_images(Mode.TEST, image_type)
    return {Mode.TRAINING: training_data, Mode.VALIDATION: validation_data, Mode.TEST: test_data}

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
    training_data = image_type.image_data[Mode.TRAINING]
    if percentage != 100:
        print(f'Selecting {percentage}% of the training data at random...')
        training_data = training_data.sample_percent(percentage)
    print('Training classifier...')
    conditionals = calc_feature_probs(image_type.categories, training_data, smoothing)
    priors = calc_priors(image_type.categories, training_data)
    print(f'Trained classifier for image type = {image_type.name}')
    return {'image_type': image_type, 'conditionals': conditionals, 'priors': priors}

def classify_naive_bayes(classifier_data, mode, indices, debug):
    """Classifies the given images using the given probability data."""
    image_data = classifier_data['image_type'].image_data[mode]
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
    if debug:
        for index, label in labels:
            print(f'Image {index} classified as: {label}')
    return labels

def train_perceptron(image_type, iterations, percentage):
    """Learns the proper weights for the Perceptron classifier
       over a given number of iterations."""
    print('Loading training data...')
    image_data = image_type.image_data[Mode.TRAINING]
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

def classify_perceptron(classifier_data, mode, indices, debug):
    """Uses the weights learned according to the Perceptron algorithm
       to classify a validation/test image."""
    image_type = classifier_data['image_type']
    image_data = image_type.image_data[mode]
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
    if debug:
        for index, label in labels:
            print(f'Image {index} classified as: {label}')
    return labels

def check_correctness(classifier_out, mode, image_type):
    """Checks how many images were correctly classified."""
    labels = image_type.image_data[mode].labels
    num_correct = 0
    total = len(classifier_out)
    for index, label in classifier_out:
        if labels[index] == label:
            num_correct += 1
    print(f'Got {num_correct} out of {total} correct: {(num_correct / total) * 100}%')

def run_classifier_bayes_smoothing(mode, image_type, indices, percentage, smoothing, debug):
    """Runs the Naive Bayes classifier from start to finish
       using a fixed, predetermined smoothing constant."""
    dat = train_naive_bayes(image_type, smoothing, percentage)
    output = classify_naive_bayes(dat, mode, indices, debug)
    check_correctness(output, mode, image_type)

def run_classifier_bayes(mode, image_type, args):
    """Runs the Naive Bayes classifier from start to finish,
       obtaining the smoothing constant from the command line args."""
    if (args.statistics):
        run_percentages_classifier("BAYES", image_type, args)
    else:
        run_classifier_bayes_smoothing(mode, image_type, range(args.range[0], args.range[1]), args.percentage, args.smoothing, args.debug)

def run_classifier_perceptron_iterations(mode, image_type, indices, percentage, iterations, debug):
    """Runs the Perceptron classifier from start to finish
       over a fixed, predetermined number of iterations."""
    dat = train_perceptron(image_type, iterations, percentage)
    output = classify_perceptron(dat, mode, indices, debug)
    check_correctness(output, mode, image_type)

def run_classifier_perceptron(mode, image_type, args):
    """Runs the Perceptron classifier from start to finish,
       obtaining the number of iterations from the command line args."""
    if (args.statistics):
        run_percentages_classifier("PERCEPTRON", image_type, args)
    else:
        run_classifier_perceptron_iterations(mode, image_type, range(args.range[0], args.range[1]), args.percentage, args.iterations, args.debug)

def run_classifier_bayes_statistics(mode, image_type, indices, percentage, smoothing, debug):
    """Runs the Naive Bayes classifier from start to finish
       using a fixed, predetermined smoothing constant."""
    dat = train_naive_bayes(image_type, smoothing, percentage)
    output = classify_naive_bayes(dat, mode, indices, debug)
    return check_correctness_statistics(output, mode, image_type)

def run_classifier_perceptron_statistics(mode, image_type, indices, percentage, iterations, debug):
    """Runs the Perceptron classifier from start to finish
       over a fixed, predetermined number of iterations."""
    dat = train_perceptron(image_type, iterations, percentage)
    output = classify_perceptron(dat, mode, indices, debug)
    return check_correctness_statistics(output, mode, image_type)

def check_correctness_statistics(classifier_out, mode, image_type):
    """Returns the percentage of images that were correctly classified."""
    labels = image_type.image_data[mode].labels
    num_correct = 0
    total = len(classifier_out)
    for index, label in classifier_out:
        if labels[index] == label:
            num_correct += 1
    return (num_correct / total) * 100

def run_percentages(debug):
    perc = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg = []
    for p in perc:
        sum = 0
        for _ in range(5):
            sum += run_classifier_bayes_statistics(Mode.TEST, ImageType.DIGIT, range(0,100), p, 2, debug)
        avg.append(sum/5)
    print(avg)

def run_percentages_classifier(classifier, image_type, args):
    perc = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg = []
    num_labels = len(image_type.image_data[Mode.TEST].labels)
    for p in perc:
        sum = 0
        for _ in range(args.statloops):
            if classifier == "BAYES":
                sum += run_classifier_bayes_statistics(Mode.TEST, image_type, range(num_labels), p, args.smoothing, args.debug)
            else:
                sum += run_classifier_perceptron_statistics(Mode.TEST, image_type, range(num_labels), p, args.iterations, args.debug)
        avg.append(sum/args.statloops)
    print(avg)

def main():
    """Command line interface for the Naive Bayes and Perceptron classifiers."""
    parser = argparse.ArgumentParser(description='Implementation of the Naive Bayes and Perceptron classifiers')
    parser.add_argument('--classifier', help='classifier to use', choices=['BAYES', 'PERCEPTRON'], required=True)
    parser.add_argument('--mode', help='image class to test', choices=['VALIDATION', 'TEST'], default='TEST')
    parser.add_argument('--type', help='image type to train', choices=['DIGIT', 'FACE'], required=True)
    parser.add_argument('--range', metavar=('START', 'END_EXCLUSIVE'), nargs=2, type=int, help='Range of data to test', default=[0, 100])
    parser.add_argument('--trainpercent', metavar='PERCENT', type=int, help='the percent of training data to use (int out of 100)', default=100, dest='percentage')
    parser.add_argument('--smoothing', type=int, help='Laplace smoothing constant (Naive Bayes)', default=2)
    parser.add_argument('--iterations', type=int, help='Number of times to iterate over training data (Perceptron)', default=5)
    parser.add_argument('--debug', help='Outputs more detailed information to stdout', action='store_true')
    parser.add_argument('--statistics', help='gathers accuracy statistics with respect to amount of training data used', action='store_true')
    parser.add_argument('--statloops', type=int, help='Number of times the classifier iterates over test data (Statistics only)', default=5)
    args = parser.parse_args()
    image_type = ImageType.DIGIT if args.type == 'DIGIT' else ImageType.FACE
    mode = Mode.TEST if args.mode == 'TEST' else Mode.VALIDATION
    run = run_classifier_bayes if args.classifier == 'BAYES' else run_classifier_perceptron
    run(mode, image_type, args)

if __name__ == '__main__':
    main()
    #run_percentages()
