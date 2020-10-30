
import pandas as pd
import numpy as np
import re
import mbrin_data_loader as data_loader
import argparse
import ujson as json
# from fancyimpute import KNN, SimpleFill, MatrixFactorization
import logging
from models.MICE import MiceImputer
import impyute.imputation.cs.mice as mice
import pandas as pd
import multiprocessing

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
def impution(idx, data):
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

    filled_mice = mice(value)
    filled_mice = np.asarray(filled_mice[np.where(eval_masks == 1)].tolist())
    mae_mi = np.abs(evals - filled_mice).mean()
    mre_mi = np.abs(evals - filled_mice).sum() / np.abs(evals).sum()

    maes_mice.append(mae_mi)
    mres_mice.append(mre_mi)
    logger.info('patient {} complex!'.format(idx))

pool = multiprocessing.Pool(processes=16)
logger.info('Processing the samples...')
for idx, data in enumerate(content):
    print('Processing patient {}'.format(idx))
    try:
        pool.apply_async(impution, (idx, data,))
    except Exception as e:
        print(e)
        continue
pool.close()
pool.join()



# print_result(maes_knn,mres_knn,'KNN')
# print_result(maes_simple,mres_simple,'Simple')
# print_result(maes_mf,mres_mf,'MF')
print_result(maes_mice,mres_mice,'MICE')
# print_result(maes_lo,mres_lo,'LO')

