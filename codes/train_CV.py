import os
import torch
import torch.nn as nn
from torch.utils import data
from metrics import analyze, calculateScore
from utils import myDataset, GradualWarmupScheduler, param_num, collate, ContrastiveLoss
from MFDm6ARice import MFDm6ARice
from train_loop import CL_Train, Validate
from data_load_processing import data_process


def crossCV(opt):
    device = opt.device
    input_path = opt.inputpath
    batch_size = opt.batch_size
    nfold = opt.nfold
    outpath = opt.outpath
    epochs = opt.epochs

    testing_result = []

    for kfold in range(nfold):

        # Load Input Representation
        train_class_fea_path = input_path + "/fold" + str(kfold) + "/train_class_fea.npy"
        valid_class_fea_path = input_path + "/fold" + str(kfold) + "/valid_class_fea.npy"
        train_class_y_path = input_path + "/fold" + str(kfold) + "/train_class_y.npy"
        valid_class_y_path = input_path + "/fold" + str(kfold) + "/valid_class_y.npy"

        train, valid, test = data_process(opt, train_class_fea_path, valid_class_fea_path,
                                          train_class_y_path, valid_class_y_path)

        train_set = myDataset(train)
        valid_set = myDataset(valid)
        test_set = myDataset(test)

        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        train_cl_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
        valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(test_set, batch_size=batch_size * 8, shuffle=False)

        model = MFDm6ARice().to(device)

        contrastive_criterion = ContrastiveLoss()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2), reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=float(200))

        best_auc = 0
        best_acc = 0
        best_epoch = 0

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        early_stopping = opt.early_stopping

        param_num(model)

        for epoch in range(epochs):
            CL_Train(model, device, train_cl_loader, contrastive_criterion, criterion, optimizer)
            train_met, _, _ = Validate(model, device, train_loader, criterion)
            valid_met, _, _ = Validate(model, device, valid_loader, criterion)
            scheduler.step()
            lr = scheduler.get_lr()[0]

            if best_acc < valid_met.acc:
                best_auc = valid_met.auc
                best_acc = valid_met.acc
                best_epoch = epoch
                path_name = os.path.join(outpath, 'model_' + str(kfold) + '.pth')
                torch.save(model, path_name)
            if epoch - best_epoch > early_stopping:
                print("Early stop at %d, %s " % (epoch, 'MFDm6ARice'))
                break
            print('{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.4f}, AUC: {:.4f} lr: {:.6f}'.format(
                'model_' + str(kfold), epoch, train_met.other[0], train_met.acc, train_met.auc, lr))

            print('{} \t Valid  Epoch: {}     avg.loss: {:.4f} Acc: {:.4f}, AUC: {:.4f} ({:.4f}) {}'.format(
                'model_' + str(kfold), epoch, valid_met.other[0], valid_met.acc, valid_met.auc, best_auc, best_epoch))

        print("{} auc: {:.4f} acc: {:.4f}".format('model_' + str(kfold), best_auc, best_acc))

        model = torch.load(path_name).to(device)
        test_met, test_label, test_score = Validate(model, device, test_loader, criterion)
        print('{} \t Test:     avg.loss: {:.4f} Acc: {:.4f}, AUC: {:.4f} ({:.4f}) {}'.format(
            'model_' + str(kfold), test_met.other[0], test_met.acc, test_met.auc, best_auc, best_epoch))

        fold_test_result = calculateScore(test_label, test_score,
                                          outpath + "/train_testy_predy_" + str(kfold + 1) + ".txt")

        testing_result.append(fold_test_result)

    temp_dict = testing_result
    analyze(temp_dict, outpath)
