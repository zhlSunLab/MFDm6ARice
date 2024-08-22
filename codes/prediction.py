import os
import torch
import torch.nn as nn
from torch.utils import data
from metrics import analyze, calculateScore
from utils import myDataset
from train_loop import Validate
from data_load_processing import data_process_pred


def pred(opt):
    device = opt.device
    input_path = opt.inputpath
    model_path = opt.model_path
    batch_size = opt.batch_size
    outpath = opt.outpath
    specie = opt.specie

    indep_testing_result = []

    data_specie = data_process_pred(input_path, specie)
    data_set = myDataset(data_specie)
    data_loader = data.DataLoader(data_set, batch_size=batch_size * 8, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2), reduction='sum')
    # model = torch.load(model_path + "MFDm6ARice.pth", map_location='cuda:0').to(device)
    model = torch.load(model_path + "MFDm6ARice.pth").to(device)

    indep_test_met, indep_test_label, indep_test_score = Validate(model, device, data_loader, criterion)
    print('Prediciton:  Acc: {:.4f}, AUC: {:.4f}'.format(indep_test_met.acc, indep_test_met.auc))

    fold_test_result = calculateScore(indep_test_label, indep_test_score, outpath + "/pred_testy_predy.txt")

    indep_testing_result.append(fold_test_result)

    temp_dict = indep_testing_result
    analyze(temp_dict, outpath)
