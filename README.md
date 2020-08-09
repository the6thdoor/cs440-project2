# cs440-project2
Intro to Machine Learning: Classifier Project

### Status
|             | Implementation     | Optimization       | Statistics                |
| :---------: | :----------------: | :----------------: | :-----------------------: |
| Naive Bayes | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:        |
| Perceptron  | :heavy_check_mark: | :x:                | :heavy_check_mark:        |

### Usage
Run `python proj2.py --classifier {BAYES, PERCEPTRON} --type {DIGIT, FACE}` to test the given classifier
on images of either digits or faces.

By default, each classifier will train using all training data, 5 iterations (Perceptron) and smoothing = 2 (Naive Bayes).
The classifiers will default to testing against the first 100 images in the test data.

To customize this, use the optional arguments `--iterations N`, `--smoothing N`, and `--range START END_EXCLUSIVE`.

For more detailed output to stdout, use `--debug`. To gather statistics, use `--statistics`.

Run `python proj2.py -h` or `python proj2.py --help` for similar information.
