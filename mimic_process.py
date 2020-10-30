# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import ujson as json
import logging
import multiprocessing





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


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

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
    task = '41401'

    save_dir = './json/'+task
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fs = open(save_dir+'/50_json', 'w')

    NUM_FEATURES = 50
    LIMIT_LEN = 64
    logger.info('Reading CSVs...')
    labels = pd.read_csv(os.path.join(data_dir, task, 'label.csv'))
    df_ori = pd.read_csv(os.path.join(data_dir, task, 'oriData.csv'))
    df_ori = df_ori.set_index('RecordID').iloc[:, 0:NUM_FEATURES]
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

    fs.close()
    print("Complete!")








