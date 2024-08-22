import numpy as np
import torch
from tqdm.auto import tqdm
import metrics as metrics


# Ref: https://github.com/LZYHZAU/PTM-CMGMS, https://awi.cuhk.edu.cn/~dbAMP/AVP/
def CL_Train(model, device, train_cl_loader, contrastive_criterion, criterion, optimizer):
    tbar = tqdm(enumerate(train_cl_loader), disable=False, total=len(train_cl_loader))
    for idx, (feature_1, feature_2, label, label1, label2) in tbar:
        model.train()
        feature_1, feature_2, label, label1, label2 = feature_1.to(device), feature_2.to(device), label.to(
            device), label1.to(device), label2.to(device)
        output1 = model(feature_1)
        output2 = model(feature_2)
        output3, _ = model.trainModel(feature_1)
        output4, _ = model.trainModel(feature_2)
        contrastive_loss = contrastive_criterion(output1, output2, label)
        class_loss_1 = criterion(output3, label1.to(torch.float32))
        class_loss_2 = criterion(output4, label2.to(torch.float32))
        loss = (contrastive_loss + class_loss_1 + class_loss_2)/3
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()


def Validate(model, device, test_loader, criterion):
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in tqdm(enumerate(test_loader), disable=False, total=len(test_loader)):
            x, y = x0.float().to(device), y0.to(device).float()
            output, _ = model.trainModel(x)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)

            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)

    met = metrics.MLMetrics()
    met.update(y_all, p_all, [l_all.mean()])

    return met, y_all, p_all
