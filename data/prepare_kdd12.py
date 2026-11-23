from data.datatransform import DataTransform, feature_example
import argparse
from pathlib import Path
import os
import pandas as pd
import tensorflow as tf
import tqdm
import sys

parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

parser.add_argument('--label', default="Label", type=str)
parser.add_argument("--store_stat", action="store_true")
parser.add_argument("--threshold", type=int, default=10)
parser.add_argument("--dataset", type=Path, default=Path("data/raw/kdd12/training.txt"))
parser.add_argument("--stats", type=Path, default=Path("data/kdd12/stats"))
parser.add_argument("--record", type=Path, default=Path("data/kdd12/record"))
parser.add_argument("--ratio", default=[0.8, 0.1, 0.1], nargs='+', type=float)

args = parser.parse_args()


class KDDTransform(DataTransform):

    def __init__(self, dataset_path, path, stats_path, min_threshold, label_index, ratio, store_stat=False, seed=2021):
        super(KDDTransform, self).__init__(dataset_path, stats_path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.split = ratio
        self.path = path
        self.stats_path = stats_path
        self.name = "Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11".split(",")

    def process(self):
        self._read(name=self.name, header=None, label_index=self.label, sep="\t")
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label)
        tr, te, val = self.random_split(ratio=self.split)
        self.transform_tfrecord_kdd(tr, self.path, "train", label_index=self.label)
        self.transform_tfrecord_kdd(te, self.path, "test", label_index=self.label)
        self.transform_tfrecord_kdd(val, self.path, "validation", label_index=self.label)

    def _process_x(self):
        print(self.data[self.data["Label"] == 1].shape)

    def _process_y(self):
        self.data["Label"] = self.data["Label"].apply(lambda x: 0 if x == 0 else 1)

    def transform_tfrecord_kdd(self, data, record_path, flag, records=5e6, label_index=""):
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

        batch_update = 1000
        while records * part <= data.shape[0]:
            tf_writer = tf.io.TFRecordWriter(os.path.join(record_path, "{}_{:04d}.tfrecord".format(flag, part)))
            print("===write part {:04d}===".format(part))
            tmp_data = data[int(part * records):int((part + 1) * records)]
            written = 0
            for index, row in tmp_data.iterrows():
                label = None
                feature = []
                # oov = True
                for i in self.field_name:
                    if i == label_index:
                        label = float(row[i])
                        continue
                    feat_id = self.feat_map.setdefault(i + "_" + str(int(row[i])), self.field_offset[i] - 1)
                    feature.append(feat_id)
                tf_writer.write(feature_example(label, feature))
                instance_num += 1
                written += 1
                if written % batch_update == 0:
                    pbar.update(batch_update)
            rem = written % batch_update
            if rem:
                pbar.update(rem)
            tf_writer.close()
            part += 1
        pbar.close()
        print("real instance number:", instance_num)


if __name__ == "__main__":
    tranformer = KDDTransform(
        args.dataset, args.record, args.stats, args.threshold, args.label, args.ratio, store_stat=args.store_stat
    )
    tranformer.process()
