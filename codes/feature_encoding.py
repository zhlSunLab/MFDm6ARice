#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : m6A_extra
@File        : feature_reproduced.py
@Author      : Mengya Liu
@Date        : 2022/10/22 17:24
@Description :

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


# file_path = "../data/NIP.both.m6A.txt"

# def read_write_file(file_path, file_out_path):
def read_write_file(file_path, file_pos_out_path, file_neg_out_path):
    # with open(file_path, 'r') as fileIN, open(file_out_path, 'w') as fileOUT:
    with open(file_path, 'r') as fileIN, open(file_pos_out_path, 'w') as filePosOUT, open(file_neg_out_path, 'w') as fileNegOUT:
        next(fileIN)
        for line in fileIN:
            line = line.strip()
            # pos_sequence = line.split('\t')[5].strip()
            # neg_sequence1 = line.split('\t')[6].strip()

            # pos_sequence = line.split('\t')[5].strip()
            # neg_sequence1 = line.split('\t')[6].strip()
            # neg_sequence2 = line.split('\t')[7].strip()

            # Maize
            pos_sequence = line.split('\t')[4].strip()
            neg_sequence1 = line.split('\t')[5].strip()
            neg_sequence2 = line.split('\t')[6].strip()

            pos_encodings = fea_cls(pos_sequence)
            neg1_encodings = fea_cls(neg_sequence1)
            neg2_encodings = fea_cls(neg_sequence2)

            # fileOUT.write("1,"+pos_encodings+",\n")
            # fileOUT.write("0,"+neg1_encodings+",\n")
            # fileOUT.write("0,"+neg2_encodings+",\n")

            filePosOUT.write("1," + pos_encodings + ",\n")
            fileNegOUT.write("0," + neg1_encodings + ",\n")
            fileNegOUT.write("0," + neg2_encodings + ",\n")


def read_write_file_pos(file_path, file_out_path):
    with open(file_path, 'r') as fileIN, open(file_out_path, 'w') as fileOUT:
        next(fileIN)
        for line in fileIN:
            line = line.strip()
            sequence = line.split(',')[0].strip()
            if 'N' not in sequence:
                encodings = fea_cls(sequence)
                fileOUT.write("1,"+encodings+",\n")


def read_write_file_neg(file_path, file_out_path):
    with open(file_path, 'r') as fileIN, open(file_out_path, 'w') as fileOUT:
        next(fileIN)
        for line in fileIN:
            line = line.strip()
            sequence = line.split(',')[0].strip()
            if 'N' not in sequence:
                encodings = fea_cls(sequence)
                fileOUT.write("0,"+encodings+",\n")


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


# file_pos_path1 = "../data/independentDataset/RFAthM6A/train_pos_ind.txt"
# file_neg_path1 = "../data/independentDataset/RFAthM6A/train_neg_ind.txt"
# file_pos_out1 = "../data/independentDataset/RFAthM6A/train_pos_ind_SMEP_fea"
# file_neg_out1 = "../data/independentDataset/RFAthM6A/train_neg_ind_SMEP_fea"
# read_write_file_pos(file_pos_path1, file_pos_out1)
# read_write_file_neg(file_neg_path1, file_neg_out1)

# file_pos_path2 = "../data/independentDataset/DeepM6ASeq-master/data/data/hs/train_pos.csv"
# file_neg_path2 = "../data/independentDataset/DeepM6ASeq-master/data/data/hs/train_neg.csv"
# file_pos_out2 = "../data/independentDataset/DeepM6ASeq-master/data/data/hs/hs_pos_SMEP_fea"
# file_neg_out2 = "../data/independentDataset/DeepM6ASeq-master/data/data/hs/hs_neg_SMEP_fea"
# read_write_file_pos(file_pos_path2, file_pos_out2)
# read_write_file_neg(file_neg_path2, file_neg_out2)
#
# file_pos_path3 = "../data/independentDataset/DeepM6ASeq-master/data/data/mm/train_pos.csv"
# file_neg_path3 = "../data/independentDataset/DeepM6ASeq-master/data/data/mm/train_neg.csv"
# file_pos_out3 = "../data/independentDataset/DeepM6ASeq-master/data/data/mm/mm_pos_SMEP_fea"
# file_neg_out3 = "../data/independentDataset/DeepM6ASeq-master/data/data/mm/mm_neg_SMEP_fea"
# read_write_file_pos(file_pos_path3, file_pos_out3)
# read_write_file_neg(file_neg_path3, file_neg_out3)

# file_test_path = r"E:\PhD_research project\NewProject20220223\m6A_extra\data\NIP.both.m6A_test.txt"
# file_test_out_path = "feature_reproduced_test_file_results"
# read_write_file(file_test_path, file_test_out_path)

# file_path = r"E:\PhD_research project\NewProject20220223\m6A_extra\data\NIP.both.m6A.txt"
# file_out_path = "feature_reproduced_file_results"
# file_pos_out_path = "feature_reproduced_file_pos_results"
# file_neg_out_path = "feature_reproduced_file_neg_results"
# read_write_file(file_path, file_out_path)
# read_write_file(file_path, file_pos_out_path, file_neg_out_path)

# file_path = r"E:\PhD_research project\NewProject20220223\m6A_extra\data\NIP.both.m6A.txt"
# # file_out_path = "feature_reproduced_file_results_balance"
# file_pos_out_path = "./20230724/feature_reproduced_file_pos_results_imbalance"
# file_neg_out_path = "./20230724/feature_reproduced_file_neg_results_imbalance"
# # read_write_file(file_path, file_out_path)
# read_write_file(file_path, file_pos_out_path, file_neg_out_path)

# file_path = "../data/maize.both.m6A.txt"
# # file_out_path = "feature_reproduced_file_results_balance"
# file_pos_out_path = "../data/independentDataset/SMEP/maize_feature_reproduced_file_pos_results"
# file_neg_out_path = "../data/independentDataset/SMEP/maize_feature_reproduced_file_neg_results"
# # read_write_file(file_path, file_out_path)
# read_write_file(file_path, file_pos_out_path, file_neg_out_path)


for i in range(5):
    file_train_path = "./fold" + str(i) + "/train_data.txt"
    file_valid_path = "./fold" + str(i) + "/valid_data.txt"
    file_train_out = "./fold" + str(i) + "/train_class_fea"
    file_valid_out = "./fold" + str(i) + "/valid_class_fea"
    read_write_file_train_valid_test(file_train_path, file_train_out)
    read_write_file_train_valid_test(file_valid_path, file_valid_out)

# file_train_path = "./train_data/train_data.txt"
# file_train_out = "./train_data/train_class_fea"
# read_write_file_train_valid_test(file_train_path, file_train_out)
#
# file_indep_test_path = "./indep_test_data/indep_test_data.txt"
# file_indep_test_out = "./indep_test_data/indep_test_class_fea"
# read_write_file_train_valid_test(file_indep_test_path, file_indep_test_out)
