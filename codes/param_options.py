import argparse


def getDefaultPara(opt):

    opt.nfold = 5               # number of cross validation
    opt.epochs = 100            # max epoch value of the model: default 100
    opt.early_stopping = 20     # early stopping patience
    opt.patience = 50           # the patience value of the early stopping
    opt.seed = 1130             # seed value of random functions

    opt.lr = 0.001              # learning rate value of the model
    opt.weight_decay = 0.       # the weight decay value of the model
    opt.batch_size = 128        # batch size of the model
    opt.max_length = 800        # max length of padding

    # opt.exp_name = 'cross'
    opt.exp_name = 'pred'
    opt.model_path = "../model/"
    opt.specie = "hs"  # os or hs

    return


def getArg():

    parser = argparse.ArgumentParser(description='MFDm6ARice for rice m6A prediction')
    parser.add_argument('-i', '--inputpath', default='../data/indeps/hs', type=str, help='the path of m6A data')
    parser.add_argument('-o', '--outpath', default='../results/indeps/hs', type=str, help='output folder')
    parser.add_argument('--exp_name', default='cross', choices=['cross_validation', 'prediction'],
                        help='experiment name of cross validation or prediction')
    parser.add_argument('-d', '--device', default='cuda:0', choices=['cpu', 'cuda', 'cuda:0'],
                        help='the device of the running environment, such as cpu.')
    parser.add_argument('--seed', default=1130, type=int, help='The random seed')
    opt = parser.parse_args()

    return opt
