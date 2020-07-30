#!/usr/bin/env python
from os import sys
from collections import namedtuple

LabeledImage = namedtuple('LabeledImage', ['data', 'label'])

# Images are 28x28.
def print_image(index, data):
    image = data[index]
    for i in range(28):
        print(image[i])

def read_image_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return [lines[i*28:(i+1)*28] for i in range(len(lines)//28)]

if __name__ == '__main__':
    print_image(int(sys.argv[1]), read_image_data("data/digitdata/testimages"))
