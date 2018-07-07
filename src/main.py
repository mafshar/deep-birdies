#!/usr/bin/env python

from processing import data_processor

DATA_DIR = './data/'

if __name__ == '__main__':
    dataloader = data_processor.load_data(DATA_DIR)
