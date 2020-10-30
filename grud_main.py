import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
import models
from models.gru_d import Model
import argparse
import impute_data_loader as data_loader
import pandas as pd
import json
import logging

from sklearn import metrics

from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='brist')
parser.add_argument('--task', type=str, default='41401')
parser.add_argument('--hid_size', type=int, default=108)
parser.add_argument('--impute_weight', type=float, default=0.3)
parser.add_argument('--label_weight', type=float, default=1.0)
parser.add_argument('--NUM_FEATURES', type=int, default=35)
parser.add_argument('--LIMIT_LEN', type=int, default=48)
args = parser.parse_args()

logger = logging.getLogger('Medical')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data_iter = data_loader.get_loader(task = args.task,batch_size=args.batch_size)

    brits_train_loss_history = []
    brits_test_loss_history = []
    brits_test_auc_history = []
    brits_test_acc_history = []
    brits_test_sum_history = []
    brits_test_mae_history = []

    roc_save,mae_save = 0,1000
    max_auc_epoch, min_mae_epoch = 0, 0

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()
            #
            logger.info('Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),)
            # print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),)

            # logger.info('Avg Loss - {}'.format(np.mean(run_loss)))
        if epoch % 1 == 0:
            acc, auc, prc, recall, precision, f1, mae, mre, val_loss = evaluate(model, data_iter)
            brits_test_loss_history.append(val_loss)
            brits_test_auc_history.append(auc)
            brits_test_acc_history.append(acc)
            brits_test_mae_history.append(mae)
            brits_test_sum_history.append(acc + auc + mae + mre)

            if auc > roc_save:
                max_auc_epoch = epoch
                roc_save = auc
            if mae < mae_save:
                min_mae_epoch = epoch
                mae_save = mae



            # print("Best AUC: %.4f" % np.max(brits_test_auc_history))
    logger.info('Best AUC - {}'.format(np.max(brits_test_auc_history)))
    logger.info('Best Acc - {}'.format(np.max(brits_test_acc_history)))
    logger.info('Best Epoch - {}'.format(np.max(brits_test_sum_history)))
    logger.info('Best AUC_Epoch - {}'.format(max_auc_epoch))
    logger.info('Best Mae_Epoch - {}'.format(min_mae_epoch))


        # evaluate(model, data_iter)


def evaluate(model, val_iter):
    model.eval()
    val_loss = 0.0

    labels = []
    preds = []
    prelabels = []

    evals = []
    imputations = []

    save_impute = []
    save_label = []

    maes = []
    mres = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        val_loss += ret['loss'].item()

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        prelabel = np.asarray((ret['predictions'].data.cpu() > 0.5).long())
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals = np.asarray(eval_[np.where(eval_masks == 1)].tolist())
        imputations = np.asarray(imputation[np.where(eval_masks == 1)].tolist())
        mae = np.abs(evals - imputations).mean()
        mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()
        maes.append(mae)
        mres.append(mre)

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]
        prelabel = prelabel[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()
        prelabels += prelabel.tolist()

    loss = val_loss / len(val_iter)

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)
    prelabels = np.asarray(prelabels)

    auc = metrics.roc_auc_score(labels, preds)
    acc = metrics.accuracy_score(labels, prelabels)
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labels, preds)
    prc = metrics.auc(recalls, precisions)
    recall = metrics.recall_score(labels, prelabels)
    precision = metrics.precision_score(labels, prelabels)
    f1 = metrics.f1_score(labels, prelabels)

    logger.info('AUC - {}'.format(auc))
    logger.info('Acc - {}'.format(acc))
    logger.info('PRC - {}'.format(prc))
    logger.info('Recall - {}'.format(recall))
    logger.info('Precision - {}'.format(precision))
    logger.info('f1 - {}'.format(f1))
    logger.info('Val Loss - {}'.format(loss))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)



    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)

    np.save('./result/{}_data'.format(args.model), save_impute)
    np.save('./result/{}_label'.format(args.model), save_label)

    maes = np.asarray(maes)
    mres = np.asarray(mres)
    mae_mean = maes.mean()
    mae_std = maes.std()
    mre_mean = mres.mean()
    mre_std = mres.std()
    logger.info('MAE(MEAN) - {}'.format(mae_mean))
    logger.info('MAE(STD) - {}'.format(mae_std))
    logger.info('MRE(MEAN) - {}'.format(mre_mean))
    logger.info('MRE(STD) - {}'.format(mre_std))
    return acc, auc, prc, recall, precision, f1, mae_mean, mre_mean, loss


def run():

    # model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight)
    logger.info('Running with args : {}'.format(args))
    model = Model(args.hid_size, args.impute_weight, args.label_weight, args.NUM_FEATURES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)


if __name__ == '__main__':
    run()

