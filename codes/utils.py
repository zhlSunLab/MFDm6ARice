import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Ref: https://github.com/LZYHZAU/PTM-CMGMS, https://awi.cuhk.edu.cn/~dbAMP/AVP/
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def shuffleData(X1, X2, y):
    index = [i for i in range(len(X1))]
    np.random.seed(0)
    np.random.shuffle(index)
    X1 = X1[index]
    X2 = X2[index]
    y = y[index]

    return X1, X2, y


class myDataset(Dataset):
    def __init__(self, data_y_list):
        self.embedding = data_y_list[0]
        self.label = data_y_list[1]

    def __getitem__(self, index):
        embedding = self.embedding[index]
        label = self.label[index]

        return embedding, label

    def __len__(self):
        return len(self.label)


def collate(batch):
    batch_a_feature = []
    batch_b_feature = []
    label_a_list = []
    label_b_list = []
    label_Comparative_learning = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        feature_a, label_a = (batch[i][0]).unsqueeze(0), (batch[i][1]).unsqueeze(0)
        feature_b, label_b = (batch[i + int(batch_size / 2)][0]).unsqueeze(0), (
            batch[i + int(batch_size / 2)][1]).unsqueeze(0)
        label_a_list.append(label_a)
        label_b_list.append(label_b)
        batch_a_feature.append(feature_a)
        batch_b_feature.append(feature_b)
        label = (label_a ^ label_b)
        label_Comparative_learning.append(label)
    feature_1 = torch.cat(batch_a_feature)
    feature_2 = torch.cat(batch_b_feature)
    label = torch.cat(label_Comparative_learning)
    label1 = torch.cat(label_a_list)
    label2 = torch.cat(label_b_list)
    return feature_1, feature_2, label, label1, label2


def param_num(model):
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("===========================")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0 - num_param1)
    print("===========================")


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
