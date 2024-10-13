#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : m6A_extra
@File        : feature_encoding.py
@Author      : Mengya Liu
@Date        : 2022/10/22 17:24
@Description : Ref: SMEP

"""
print(__doc__)

chemical_property = {
    'A': "1 ",
    'T': "2 ",
    'G': "3 ",
    'C': "4 ",
}


def fea_cls(sequence):
    encodings = ""
    for base in sequence:
        encodings += chemical_property[base]
    return encodings


def read_write_file_train_valid_test(file_path, file_out_path):
    with open(file_path, 'r') as fileIN, open(file_out_path, 'w') as fileOUT:
        next(fileIN)
        for line in fileIN:
            line = line.strip()
            sequence = line.split(',')[0].strip()
            label = line.split(',')[1].strip()
            if 'N' not in sequence:
                encodings = fea_cls(sequence)
                fileOUT.write(label + "," + encodings+",\n")


# Example
file_train_path = "./train_data/train_data.txt"
file_train_out = "./train_data/train_class_fea"
read_write_file_train_valid_test(file_train_path, file_train_out)
