import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, roc_auc_score, auc, \
    precision_recall_curve, accuracy_score

__all__ = [
    "accuracy",
    "roc",
    "pr",
    "calculate_metrics",
    "calculateScore",
    "analyze"
]


class MLMetrics(object):
    def __init__(self, objective='binary'):
        self.objective = objective
        self.metrics = []

    def update(self, label, pred, other_lst):
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst) > 0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg()

    def compute_avg(self):
        if len(self.metrics) > 1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.auc = self.avg[1]
        self.prc = self.avg[2]
        self.tp = int(self.sum[3])
        self.tn = int(self.sum[4])
        self.fp = int(self.sum[5])
        self.fn = int(self.sum[6])
        if len(self.avg) > 7:
            self.other = self.avg[7:]


def accuracy(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(accuracy_score(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros(num_labels)
        for i in range(num_labels):
            metric[i] = accuracy_score(label[:, i], np.round(prediction[:, i]))
    return metric


def roc(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, thresholds = roc_curve(label, prediction)
        score = auc(fpr, tpr)
        metric = np.array(score)
        curves = [(fpr, tpr)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros(num_labels)
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:, i], prediction[:, i])
            score = auc(fpr, tpr)
            metric[i] = score
            curves.append((fpr, tpr))
    return metric, curves


def pr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        score = auc(recall, precision)
        metric = np.array(score)
        curves = [(precision, recall)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros(num_labels)
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:, i], prediction[:, i])
            score = auc(recall, precision)
            metric[i] = score
            curves.append((precision, recall))
    return metric, curves


def tfnp(label, prediction):
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn = 0, 0, 0, 0

    return tp, tn, fp, fn


def calculate_metrics(label, prediction, objective):
    if objective == "binary":
        ndim = np.ndim(label)

        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)

        if ndim == 2:
            prediction = prediction[:, 0]
            label = label[:, 0]

        pred_class = prediction > 0.5

        tp, tn, fp, fn = tfnp(label, pred_class)

        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

    else:
        mean = 0
        std = 0

    return [mean, std]


def calculateScore(y, pred_y, OutputFile):
    with open(OutputFile, 'w') as fOUT:
        for index in range(len(y)):
            fOUT.write(str(y[index]) + '\t' + str(pred_y[index]) + '\n')

    tempLabel = [(0 if i < 0.5 else 1) for i in pred_y]
    confusion = confusion_matrix(y, tempLabel)

    TN, FP, FN, TP = confusion.ravel()
    MCC = matthews_corrcoef(y, tempLabel)
    accuracy = (TP + TN) / float(len(y))

    ROCArea = roc_auc_score(y, pred_y)

    precisionPR, recallPR, _ = precision_recall_curve(y, pred_y)
    aupr = auc(recallPR, precisionPR)

    return {'acc': accuracy, 'MCC': MCC, 'AUC': ROCArea, 'AUPR': aupr}


def analyze(temp, OutputDir):
    testing_result = temp

    # The performance output file about test set
    file = open(OutputDir + '/performance.txt', 'w')
    for x in [testing_result]:
        title = 'testing_'
        file.write(title + 'results\n')
        for j in ['acc', 'MCC', 'AUC', 'AUPR']:
            total = []
            for val in x:
                total.append(val[j])
            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total)) + '\n')
        file.write('\n\n______________________________\n')
    file.close()
