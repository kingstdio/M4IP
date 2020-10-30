
import pandas as pd
import numpy as np
import re
import mbrin_data_loader as data_loader
import argparse
import json
from fancyimpute import KNN, SimpleFill, MatrixFactorization, IterativeImputer, NuclearNormMinimization
import logging
from models.MICE import MiceImputer
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--task', type=str, default='25000')
args = parser.parse_args()

logger = logging.getLogger('Medical')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

task = args.task
data_dir = './json/mbrin/'+ task +'/impute_json'
content = open(data_dir).readlines()
indices = np.arange(len(content))
val_indices = np.random.choice(indices, len(content) // 5)

val_indices = set(val_indices.tolist())




def print_result(maes_,mres_,method):
    logger.info(method)
    maes = np.asarray(maes_)
    mres = np.asarray(mres_)
    mae_mean = np.nanmean(maes)
    mae_std =  np.nanstd(maes)
    mre_mean = np.nanmean(mres)
    mre_std = np.nanstd(mres)
    logger.info('MAE(MEAN) - {}'.format(mae_mean))
    logger.info('MAE(STD) - {}'.format(mae_std))
    logger.info('MRE(MEAN) - {}'.format(mre_mean))
    logger.info('MRE(STD) - {}'.format(mre_std))


maes_knn = []
mres_knn = []
maes_simple = []
mres_simple = []
maes_mf = []
mres_mf = []
maes_mice = []
mres_mice = []
maes_lo = []
mres_lo = []
maes_mean = []
mres_mean = []
maes_ii = []
mres_ii = []
maes_nnm = []
mres_nnm = []




def do_impute(idx, data):
    logger.info('Processing patient {}'.format(idx))
    rec = json.loads(content[idx])
    rec = rec['forward']
    value = np.array(rec['values'])
    # a = np.array(value)
    eval_ = np.array(rec['evals'])
    eval_masks = np.array(rec['eval_masks'])
    masks = np.array(rec['masks'])
    forwards = np.array(rec['forwards'])
    value[np.where(masks == 0)] = np.nan

    value = pd.DataFrame(value)
    evals = pd.DataFrame(eval_)
    eval_masks = pd.DataFrame(eval_masks)
    col_name = value.columns.values.tolist()
    for item in col_name:
        if value[item].isna().all():
            value = value.drop([item], axis=1)
            eval_masks = eval_masks.drop([item], axis=1)
            evals = evals.drop([item], axis=1)
    value = value.values
    evals = evals.values
    eval_masks = eval_masks.values

    evals = np.asarray(eval_[np.where(eval_masks == 1)].tolist())



    filled_ii = IterativeImputer().fit_transform(value)
    filled_ii = np.asarray(filled_ii[np.where(eval_masks == 1)].tolist())
    mae_i = np.abs(evals - filled_ii).mean()
    mre_i = np.abs(evals - filled_ii).sum() / np.abs(evals).sum()
    maes_ii.append(mae_i)
    mres_ii.append(mre_i)
    # print('II:',filled_ii)
#    logger.info('MAE: {}'.format(mae_i))
#    logger.info('MRE: {}'.format(mre_i))
#    logger.info('maes_ii: {}'.format(maes_ii))
#    logger.info('mres_ii: {}'.format(mres_ii))

    filled_nnm = NuclearNormMinimization().fit_transform(value)
    filled_nnm = np.asarray(filled_nnm[np.where(eval_masks == 1)].tolist())
    mae_n = np.abs(evals - filled_nnm).mean()
    mre_n = np.abs(evals - filled_nnm).sum() / np.abs(evals).sum()
    maes_nnm.append(mae_n)
    mres_nnm.append(mre_n)

#    logger.info('MAE: {}'.format(mae_n))
#    logger.info('MRE: {}'.format(mre_n))
#    logger.info('maes_nnm: {}'.format(maes_nnm))
#    logger.info('mres_nnm: {}'.format(mres_nnm))





for idx, data in enumerate(content):
    if idx in val_indices:
        do_impute(idx, data,)






# print_result(maes_knn,mres_knn,'KNN')
# print_result(maes_simple,mres_simple,'Simple')
# print_result(maes_mf,mres_mf,'MF')
# print_result(maes_mice,mres_mice,'MICE')
# print_result(maes_lo,mres_lo,'LO')
# print_result(maes_mean,mres_mean,'MEAN')

print_result(maes_ii,mres_ii,'IterativeImputer')
print_result(maes_nnm,mres_nnm,'NuclearNormMinimization')


