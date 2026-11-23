import os.path
import tensorflow as tf
import numpy as np
import os
import sys
from collections import defaultdict
import math
from pathlib import Path
import shutil
import pickle
import tqdm
import pandas as pd
from sklearn.utils import shuffle
from abc import abstractmethod


def feature_example(label, feature):
    feature_des = {
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=feature))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_des))
    return example_proto.SerializeToString()


class DataTransform(object):

    def __init__(self, dataset_path, stats_path, store_stat=False, seed=2021):
        self.data_path = dataset_path
        self.store_stat = store_stat
        self.seed = seed
        self.stats_path = stats_path
        self.feat_map = {}
        self.defaults = {}

        # container to hold per-field counts for saved features
        self.save_cnt = {}

        if store_stat:
            os.makedirs(self.stats_path, exist_ok=True)
        else:
            with open(self.stats_path.joinpath("feat_map.pkl"), 'rb') as fi:
                self.feat_map = pickle.load(fi)
            with open(self.stats_path.joinpath("defaults.pkl"), 'rb') as fi:
                self.defaults = pickle.load(fi)
            with open(self.stats_path.joinpath("offset.pkl"), 'rb') as fi:
                self.field_offset = pickle.load(fi)
            # attempt to load previously saved counts (optional)
            save_cnt_path = self.stats_path.joinpath("save_cnt.pkl")
            if save_cnt_path.exists():
                with open(save_cnt_path, 'rb') as fi:
                    self.save_cnt = pickle.load(fi)

    def _read(self, name=None, header=None, sep=None, label_index=""):
        print("=====read data=====")

        self.data = pd.read_table(self.data_path, names=name, header=header, sep=sep)

        print(self.data)
        self._process_x()
        self._process_y()
        print(self.data)

        self.num_instances = self.data.shape[0]
        self.num_fields = self.data.shape[1] - 1

        self.field_name = self.data.columns.values.tolist()
        assert self.num_fields == len(self.field_name) - 1

        num_features = 0
        for field in self.field_name:
            if field == label_index:
                continue
            num_features += self.data[field].unique().size
        print("===Data summary===")
        print("instances:{}, fields:{}, raw_features:{}".format(self.num_instances, self.num_fields, num_features))

    def generate_and_filter(self, threshold=0, label_index="", white_list=[]):
        self.field_offset = {}
        offset = 0
        for field in self.field_name:
            #print(field)
            if field == label_index:
                continue
            feat_count = self.data[field].value_counts(dropna=False).to_dict()
            if field not in white_list:
                unique_feat = [key for key, value in feat_count.items() if value >= threshold]
                default_list = [value for key, value in feat_count.items() if value < threshold]
            else:
                unique_feat = [key for key, value in feat_count.items()]
                default_list = []

            field_feat_map = dict((field + "_" + str(j), i + offset) for i, j in enumerate(unique_feat))
            self.feat_map.update(field_feat_map)

            if len(feat_count) == len(unique_feat):
                offset += len(unique_feat)
            else:
                offset += len(unique_feat) + 1

            self.defaults.update({field: len(unique_feat)})
            self.field_offset.update({field: offset})

            # prepare and save counts mapping: index -> frequency
            cnt = {i: feat_count[j] for i, j in enumerate(unique_feat)}
            if default_list:
                cnt[len(unique_feat)] = sum(default_list)
            self.save_cnt.update({field: cnt})
        print("After filtering features:{}".format(len(self.feat_map)))
        print("Final offset:{}".format(offset))

        with open(self.stats_path.joinpath("feat_map.pkl"), 'wb') as fi:
            pickle.dump(self.feat_map, fi)
        with open(self.stats_path.joinpath("defaults.pkl"), 'wb') as fi:
            pickle.dump(self.defaults, fi)
        with open(self.stats_path.joinpath("offset.pkl"), 'wb') as fi:
            pickle.dump(self.field_offset, fi)
        with open(self.stats_path.joinpath("save_cnt.pkl"), 'wb') as fi:
            pickle.dump(self.save_cnt, fi)

    def random_split(self, ratio=[]):
        assert len(ratio) == 3, "give three dataset ratio"
        train_data = self.data.sample(frac=ratio[0], replace=False, axis=0, random_state=self.seed)
        left_data = self.data[~self.data.index.isin(train_data.index)]
        val_data = left_data.sample(frac=ratio[1] / (ratio[1] + ratio[2]), replace=False, axis=0, random_state=self.seed)
        test_data = left_data[~left_data.index.isin(val_data.index)]
        print("===Train size:{}===".format(train_data.shape[0]))
        print("===Test size:{}===".format(test_data.shape[0]))
        print("===Validation size:{}===".format(val_data.shape[0]))
        return train_data, val_data, test_data

    def transform_tfrecord(self, data, record_path, flag, records=5e6, label_index=""):
        os.makedirs(record_path, exist_ok=True)
        part = 0
        instance_num = 0
        # Use a single global progress bar. When stderr is not a TTY (e.g. nohup >> log),
        # disable tqdm to avoid printing many lines to the log file.
        total_instances = int(data.shape[0])
        use_tqdm = sys.stderr.isatty()
        pbar = tqdm.tqdm(
            total=total_instances,
            unit='samples',
            bar_format='{l_bar}{bar} {percentage:3.0f}% [{elapsed}<{remaining}]',
            ncols=80,
            mininterval=1.0,
            disable=not use_tqdm,
            leave=True,
        )

        # update in small batches to reduce overhead (if desired)
        batch_update = 1000
        while records * part <= data.shape[0]:
            tf_writer = tf.io.TFRecordWriter(os.path.join(record_path, "{}_{:04d}.tfrecord".format(flag, part)))
            print("===write part {:04d}===".format(part))
            tmp_data = data[int(part * records):int((part + 1) * records)]
            written = 0
            for index, row in tmp_data.iterrows():
                label = None
                feature = []
                for i in self.field_name:
                    if i == label_index:
                        label = float(row[i])
                        continue
                    feat_id = self.feat_map.setdefault(i + "_" + str(row[i]), self.field_offset[i] - 1)
                    feature.append(feat_id)
                tf_writer.write(feature_example(label, feature))
                instance_num += 1
                written += 1
                # update progress bar (no-op if disabled)
                if written % batch_update == 0:
                    pbar.update(batch_update)
            # update any remaining
            rem = written % batch_update
            if rem:
                pbar.update(rem)
            tf_writer.close()
            part += 1
        pbar.close()
        print("real instance number:", instance_num)

    @abstractmethod
    def _process_x(self):
        pass

    @abstractmethod
    def _process_y(self):
        pass

    @abstractmethod
    def process(self):
        pass
