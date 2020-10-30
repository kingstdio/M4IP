
import pandas as pd
import numpy as np
import re
import mbrin_data_loader as data_loader
import argparse
import ujson as json
from fancyimpute import KNN, SimpleFill, MatrixFactorization
import logging
from models.MICE import MiceImputer
import impyute.imputation.cs.mice as mice
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--task', type=str, default='41401')
args = parser.parse_args()

logger = logging.getLogger('Medical')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

task = args.task
data_dir = './json/mbrin/'+ task +'/50_json'
content = open(data_dir).readlines()
a = content



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


maes_mice = []
mres_mice = []
maes_mf = []
mres_mf = []
for idx, data in enumerate(content):
    rec = json.loads(content[idx])
    rec = rec['forward']
    value = np.array(rec['values'])
    # a = np.array(value)
    eval_ = np.array(rec['evals'])
    eval_masks = np.array(rec['eval_masks'])
    masks = np.array(rec['masks'])
    forwards = np.array(rec['forwards'])
    value[np.where(masks==0)] = np.nan



    # filled_knn = KNN(k=3).fit_transform(value)
    # filled_knn = np.asarray(filled_knn[np.where(eval_masks == 1)].tolist())
    # mae_k = np.abs(evals - filled_knn).mean()
    # mre_k = np.abs(evals - filled_knn).sum() / np.abs(evals).sum()
    # maes_knn.append(mae_k)
    # mres_knn.append(mre_k)
    #
    # filled_simple = SimpleFill().fit_transform(value)
    # filled_simple = np.asarray(filled_simple[np.where(eval_masks == 1)].tolist())
    # mae_s = np.abs(evals - filled_simple).mean()
    # mre_s = np.abs(evals - filled_simple).sum() / np.abs(evals).sum()
    # maes_simple.append(mae_s)
    # mres_simple.append(mre_s)

    # filled_mf = MatrixFactorization().fit_transform(value)
    # filled_mf = np.asarray(filled_mf[np.where(eval_masks == 1)].tolist())
    # mae_m = np.abs(evals - filled_mf).mean()
    # mre_m = np.abs(evals - filled_mf).sum() / np.abs(evals).sum()
    # maes_mf.append(mae_m)
    # mres_mf.append(mre_m)

    # filled_mice = MiceImputer().fit_transform(pd.DataFrame(value))
    # n = 5
    # arr = np.random.uniform(high=6, size=(n, n))
    # for _ in range(3):
    #     arr[np.random.randint(n), np.random.randint(n)] = np.nan
    # # filled_arr = mice(arr)
    # print(type(arr))
    # print(type(value))
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


    # filled_mice = mice(value)
    # filled_mice = np.asarray(filled_mice[np.where(eval_masks == 1)].tolist())
    # mae_mi = np.abs(evals - filled_mice).mean()
    # mre_mi = np.abs(evals - filled_mice).sum() / np.abs(evals).sum()
    # maes_mice.append(mae_mi)
    # mres_mice.append(mre_mi)

    filled_mf = MatrixFactorization().fit_transform(value)
    filled_mf = np.asarray(filled_mf[np.where(eval_masks == 1)].tolist())
    mae_m = np.abs(evals - filled_mf).mean()
    mre_m = np.abs(evals - filled_mf).sum() / np.abs(evals).sum()
    maes_mf.append(mae_m)
    mres_mf.append(mre_m)

    # filled_lo = forwards
    # filled_lo = np.asarray(filled_lo[np.where(eval_masks == 1)].tolist())
    # mae_l = np.abs(evals - filled_lo).mean()
    # mre_l = np.abs(evals - filled_lo).sum() / np.abs(evals).sum()
    # maes_lo.append(mae_l)
    # mres_lo.append(mre_l)



# print_result(maes_knn,mres_knn,'KNN')
# print_result(maes_simple,mres_simple,'Simple')
print_result(maes_mf,mres_mf,'MF')
# print_result(maes_mice,mres_mice,'MICE')
# print_result(maes_lo,mres_lo,'LO')

