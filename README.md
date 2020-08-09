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

For more detailed output to stdout, add the `--debug` flag. To gather statistics, use `--statistics` flag.
Note that `--statistics` will ignore the range information and automatically test against all testing data.

Examples:
```
python proj2.py --classifier PERCEPTRON --type DIGIT --statistics
Gathers statistics for the Perceptron classifier trained for digits (with default iterations = 5).

python proj2.py --classifier BAYES --type FACE --debug
Gathers statistics for the Naive Bayes classifier trained for faces (with default smoothing = 2).

python proj2.py --classifier BAYES --type DIGIT --smoothing 10 --range 0 1000 --debug
Naive Bayes, on digits, with smoothing = 10, testing against all test data, with debug output.

python proj2.py --classifier PERCEPTRON --type FACE --iterations 20 --range 0 150
Perceptron, on faces, with iterations = 20, testing against all test data.

python proj2.py --classifier PERCEPTRON --type DIGIT --statistics --statloops 50
Perceptron, on digits, computing accuracy statistics by averaging 50 trials (default is 5).
```

Run `python proj2.py -h` or `python proj2.py --help` for similar information.
