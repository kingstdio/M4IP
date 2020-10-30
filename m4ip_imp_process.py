# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import ujson as json
import logging
import multiprocessing
import argparse
from fancyimpute import KNN, SimpleFill


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='5849')
parser.add_argument('--NUM_FEATURES', type=int, default=50)
parser.add_argument('--LIMIT_LEN', type=int, default=64)
args = parser.parse_args()


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(LIMIT_LEN):
        if h == 0:
            deltas.append(np.ones(NUM_FEATURES))
        else:
            deltas.append(np.ones(NUM_FEATURES) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_bursty(values):
    bursty = []
    for i in range(NUM_FEATURES):
        u = np.nanmean(values[:,i])
        s = np.nanstd(values[:,i])
        b = (s - u) / (s + u)
        if np.isnan(b):
            b = 0
        bursty.append(b)
    # temp = bursty + np.zeros(values.shape)
    bursty += np.zeros(values.shape)

    return np.array(bursty)

def parse_mrate(masks):
    mrates = []
    for i in range(NUM_FEATURES):
        mrate = []
        for j in range(LIMIT_LEN):
            miss = sum(masks[:j+1,i]==0)
            rate = [miss/(j+1)]
            mrate.append(rate)
        if i == 0:
            mrates = mrate.copy()
        else:
            mrates = np.concatenate((mrates,mrate), axis=1)

    return np.array(mrates)






def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)
    bursty = parse_bursty(values)
    mrates = parse_mrate(masks)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()
    knn = KNN(k=3).fit_transform(values)
    mean = SimpleFill().fit_transform(values)

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()
    rec['bursty'] = bursty.tolist()
    rec['mrates'] = mrates.tolist()

    rec['knn'] = knn.tolist()
    rec['mean'] = mean.tolist()
    return rec


def parse_id(record_id, label):
    # data = pd.read_csv('./raw/{}.txt'.format(id_))
    # # accumulate the records within one hour
    # data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))

    x_ori = df_ori.loc[[record_id]].as_matrix()
    sample_len = len(x_ori)
    if sample_len < LIMIT_LEN:
        fill =  np.full([LIMIT_LEN - sample_len, NUM_FEATURES], np.nan)
        x_ori = np.concatenate((x_ori, fill), axis=0)
    else:
        x_ori = x_ori[0:LIMIT_LEN]

    evals = x_ori
    # evals = (np.array(evals) - mean) / std

    shp = evals.shape

    evals = evals.reshape(-1)

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10)

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    # label = int(out.loc[int(id_)])

    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    rec = json.dumps(rec)

    fs.write(rec + '\n')


if __name__ == '__main__':
    logger = logging.getLogger('Medical')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    data_dir = '../Data/'
    task = args.task

    save_dir = './json/mbrin/'+task
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fs = open(save_dir+'/impute_json', 'w')

    NUM_FEATURES = args.NUM_FEATURES
    LIMIT_LEN = args.LIMIT_LEN
    logger.info('Reading CSVs...')
    labels = pd.read_csv(os.path.join(data_dir, task, 'label.csv'))
    df_ori = pd.read_csv(os.path.join(data_dir, task, 'oriData.csv'))
    df_ori = df_ori.set_index('RecordID').iloc[:, 0:NUM_FEATURES]
    mean = df_ori.loc[labels['icustay_id']].iloc[:, 0:].mean().values
    std = df_ori.loc[labels['icustay_id']].iloc[:, 0:].std().values
    # data_ori = df_ori.loc[labels['icustay_id']]
    pool = multiprocessing.Pool(processes=16)
    logger.info('Processing the samples...')
    for i, (record_id, label) in enumerate(zip(labels['icustay_id'], labels['isAlive'])):
        print('Processing patient {}'.format(record_id))
        try:
            pool.apply_async(parse_id, (record_id, label,))
        except Exception as e:
            print(e)
            continue
    pool.close()
    pool.join()
    # for i, (record_id, label) in enumerate(zip(labels['icustay_id'], labels['isAlive'])):
    #     parse_id(record_id, label)
    fs.close()
    print("Complete!")








