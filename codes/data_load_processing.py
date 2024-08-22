import numpy as np
import torch


def shuffleData(X1, y):
    index = [i for i in range(len(X1))]
    np.random.seed(0)
    np.random.shuffle(index)
    X1 = X1[index]
    y = y[index]

    return [torch.tensor(X1), torch.LongTensor(y)]


def data_process(opt, train_class_fea_path, valid_class_fea_path, train_class_y_path, valid_class_y_path):
    train_class_fea = np.load(train_class_fea_path)
    valid_class_fea = np.load(valid_class_fea_path)
    test_class_fea = np.load(valid_class_fea_path)

    train_class_y = np.load(train_class_y_path).astype(np.float32).reshape(-1, 1)
    valid_class_y = np.load(valid_class_y_path).astype(np.float32).reshape(-1, 1)
    test_class_y = np.load(valid_class_y_path).astype(np.float32).reshape(-1, 1)

    train = shuffleData(train_class_fea, train_class_y)
    valid = shuffleData(valid_class_fea, valid_class_y)
    test = shuffleData(test_class_fea, test_class_y)

    return train, valid, test


def data_process_pred(class_data_y_path, specie):
    global data_X, data_y

    if specie == "os":
        data_X = np.load(class_data_y_path + '/os_same_species_indep_test_class_fea.npy')
        data_y = np.load(class_data_y_path + '/os_same_species_indep_test_class_y.npy').astype(np.float32).reshape(-1,
                                                                                                                   1)

    if specie == "hs":
        pos_data_class_fea = np.load(class_data_y_path + '/hs_pos_cross_species_indep_test_class_fea.npy')
        neg_data_class_fea = np.load(class_data_y_path + '/hs_neg_cross_species_indep_test_class_fea.npy')

        pos_data_class_y = np.load(
            class_data_y_path + '/hs_pos_cross_species_indep_test_class_y.npy').astype(np.float32).reshape(-1, 1)
        neg_data_class_y = np.load(
            class_data_y_path + '/hs_neg_cross_species_indep_test_class_y.npy').astype(np.float32).reshape(-1, 1)

        data_X = np.concatenate((pos_data_class_fea, neg_data_class_fea))
        data_y = np.concatenate((pos_data_class_y, neg_data_class_y))

    return [torch.tensor(data_X), torch.LongTensor(data_y)]
