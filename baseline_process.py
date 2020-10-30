import pandas as pd
import numpy as np
import os
import pickle as pkl
import json
from tqdm import tqdm
import argparse
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='5849')
parser.add_argument('--NUM_FEATURES', type=int, default=50)
parser.add_argument('--LIMIT_LEN', type=int, default=64)
args = parser.parse_args()


def convert_samples(samples):
    print('Converting samples...')
    indexes = samples[0]['index']
    labels = [samples[0]['label']] * len(indexes)
    del samples[0]
    for sample in tqdm(samples):
        # indexes = np.vstack((indexes, sample['index']))
        indexes = np.concatenate((indexes, sample['index']), axis=0)
        labels += [sample['label']] * len(sample['index'])

    labels = np.asarray(labels)
    return indexes, labels


def get_data(outcomes,df_ori):
    samples, labels = [],[]
    for i, (record_id, label) in enumerate(zip(outcomes['icustay_id'], outcomes['isAlive'])):
        x_ori = df_ori.loc[[record_id]].as_matrix()

        sample_len = len(x_ori)
        if sample_len < args.LIMIT_LEN:
            fill = np.full([args.LIMIT_LEN - sample_len, args.NUM_FEATURES], np.nan)
            x_ori = np.concatenate((x_ori, fill), axis=0)
        else:
            x_ori = x_ori[0:args.LIMIT_LEN]
        x_ori = pd.DataFrame(x_ori).fillna(method='ffill').fillna(0.0).as_matrix()
        x_ori = x_ori.reshape(-1)
        samples.append(x_ori.tolist())
    labels = outcomes['isAlive'].tolist()
    return samples,labels



def preprocess_data(train_outcomes, test_outcomes,df_ori):
    train_samples,train_labels = get_data(train_outcomes,df_ori)
    test_samples,test_labels = get_data(test_outcomes,df_ori)
    return np.array(train_samples, dtype=np.float32), np.array(train_labels, dtype=np.int32), np.array(test_samples, dtype=np.float32), np.array(test_labels, dtype=np.int32)


def save(data, file_name, data_type):
    print('Saving {} data..c.'.format(data_type))
    np.save(file_name, data)
    # with open(file_name, 'w') as f:
    #     json.dump(data, f)
    # f.close()


def run(path,subtask,maintask,sets):
    train_x, train_y, test_x, test_y = preprocess_data(train_outcomes, test_outcomes,df_ori)
    print('train_x:', train_x.shape, 'train_y:', train_y.shape)
    out_path = 'data/preprocessed_data/baseline/' + subtask
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save(train_x, 'data/preprocessed_data/baseline/' + subtask + '/train_x.npy', 'train_x')
    save(train_y, 'data/preprocessed_data/baseline/' + subtask + '/train_y.npy', 'train_y')
    save(test_x, 'data/preprocessed_data/baseline/' + subtask + '/test_x.npy', 'test_x')
    save(test_y, 'data/preprocessed_data/baseline/' + subtask + '/test_y.npy', 'test_x')
    print(subtask+"Complete!")

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
    outcomes = pd.read_csv(os.path.join(data_dir, task, 'label.csv'))
    df_ori = pd.read_csv(os.path.join(data_dir, task, 'oriData.csv'))
    df_ori = df_ori.set_index('RecordID').iloc[:, 0:args.NUM_FEATURES]
    train_outcomes, test_outcomes = train_test_split(outcomes, test_size=0.2)
    train_x, train_y, test_x, test_y = preprocess_data(train_outcomes, test_outcomes,df_ori)
    print('train_x:', train_x.shape, 'train_y:', train_y.shape)
    out_path = 'data/preprocessed_data/baseline/' + task
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save(train_x, 'data/preprocessed_data/baseline/' + task + '/train_x.npy', 'train_x')
    save(train_y, 'data/preprocessed_data/baseline/' + task + '/train_y.npy', 'train_y')
    save(test_x, 'data/preprocessed_data/baseline/' + task + '/test_x.npy', 'test_x')
    save(test_y, 'data/preprocessed_data/baseline/' + task + '/test_y.npy', 'test_x')
    print(task," Complete!")
