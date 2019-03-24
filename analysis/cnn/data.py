"""
Data loader

might only use from_generator for simplicity

main author: Jeff
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np

from utils import load_img, resample_img

root = "D:\\"
DATA_DIR = os.path.join(root, "fmri", "ABCD", "data")

N_THREADS = 4

def create_testing_generator(shape):
    def gen():
        while True:
            sample, label = np.random.rand(*shape), round(np.random.rand())

            yield (sample, label)
    return gen


def nifti_generator(file_list, nifti_path, labels, vol_labels, voxel_size_mm=None):
    """ loads using nibabel"""
    with open(file_list, "r") as f:
        subjects = [s.strip() for s in f.readlines()]

    volume_idx = ["_volume" in i for i in vol_labels.columns]

    ### probably important ###
    entropy_idx = ["_entropy" in i for i in vol_labels.columns]

    ### THESE FEATURE PROBABLY WON'T BE USED, CAN BE INFERRED BY CNN ###
    mean_idx = ["_intmean" in i for i in vol_labels.columns]
    stdev_idx = ["_intstdev" in i for i in vol_labels.columns]

    for s in subjects:
        # TODO
        try:
            # tf.logging.info(s)
            vol_features = vol_labels.loc[s, volume_idx].values
            ent_features = vol_labels.loc[s, entropy_idx].values
            mean_features = vol_labels.loc[s, mean_idx].values
            stdev_features = vol_labels.loc[s, stdev_idx].values
            label = labels.loc[s]["residual_fluid_intelligence_score"] if s in labels else 0
            im = load_img(os.path.join(nifti_path, s, "baseline", "structural", "t1_brain.nii.gz")) #  os.path.join(nifti_path, s, b"baseline", b"structural", b"t1_brain.nii.gz"))
            if voxel_size_mm:
                target_affine = np.diag((voxel_size_mm, voxel_size_mm, voxel_size_mm))
                im = resample_img(im, target_affine)
            yield im.get_data(), vol_features, ent_features, mean_features, stdev_features, label
        except (KeyError, FileNotFoundError):
            tf.logging.info("{} not loaded".format(s))
            continue

""" utils """

# dataset = dataset.map(lambda x: x)
def _preprocess(x, crop_size):
    image = x["image"]  # x, y, z
    s = tf.shape(image)
    center = s / 2
    image = image[
        tf.cast(tf.floor(center[0]), tf.int32) - crop_size[0] // 2:tf.cast(tf.ceil(center[0]), tf.int32) + crop_size[0] // 2,
        tf.cast(tf.floor(center[1]), tf.int32) - crop_size[1] // 2:tf.cast(tf.ceil(center[1]), tf.int32) + crop_size[1] // 2,
        tf.cast(tf.floor(center[2]), tf.int32) - crop_size[2] // 2:tf.cast(tf.ceil(center[2]), tf.int32) + crop_size[2] // 2
    ]

    s = tf.shape(image)
    x["image"] = tf.reshape(image, [s[0], s[1], s[2], 1])
    return x

def _resize(x):
    image = x["image"]
    s = tf.shape(image)
    x["image"] = tf.reshape(image, [s[0], s[1], s[2], 1])
    return x

def _load_features():
    vol_labels = pd.concat(
        [pd.read_excel(
            os.path.join(DATA_DIR, "results", "ALL_measures_evan.xlsx"),
            header=0,
            skiprows=1),
         pd.read_excel(
            os.path.join(DATA_DIR, "results", "ALL_measures_evan_validation.xlsx"),
            header=0,
            skiprows=1),
         pd.read_excel(
            os.path.join(DATA_DIR, "results", "ALL_measures_evan_testing.xlsx"),
            header=0,
            skiprows=1),
        ]).replace("missing", 0).rename(columns={"ID": "subject"}).set_index("subject", drop=True).fillna(0)
    for idx in np.where(vol_labels.dtypes == 'O')[0]:
        vol_labels.iloc[:, idx] = vol_labels.iloc[:, idx].map(float)
    return vol_labels

class DataLoader():
    def __init__(self, cfg=None, nifti_path=None, features_path=None, labels_path=None):

        self.cfg=cfg if cfg else {}
        self.cfg["batch_size"] = 4
        self.cfg["max_concurrent_files"] = 4
        self.cfg["num_interleave"] = 16
        self.cfg["rs_voxel_size"] = 2
        self.cfg["crop_size"] = [79, 91, 79]
        self.cfg["nifti_path"] = os.path.join(DATA_DIR, "fmriresults01", "image03")

        self.labels = pd.concat(
            [
                pd.read_csv(
                    os.path.join(DATA_DIR,
                                 "results",
                                 "training_fluid_intelligenceV1.csv")
                    ),
                pd.read_csv(
                    os.path.join(DATA_DIR,
                                 "results",
                                 "validation_fluid_intelligenceV1.csv")
                    )
            ]).set_index("subject")

        tf.logging.info("Loading ROI features...")
        self.vol_labels = _load_features()


        self.feature_size = {
            "volume": sum(["_volume" in i for i in self.vol_labels.columns]),
            "entropy": sum(["_entropy" in i for i in self.vol_labels.columns]),
            "mean": sum(["_intmean" in i for i in self.vol_labels.columns]),
            "stdev": sum(["_intstdev" in i for i in self.vol_labels.columns])
        }

        tf.logging.info("done")

    def input_fn(self):
        pass

    def augment(self, dataset, batch_size=4):
        """ data augmentation here """
        def _augment(x):
            return x
        dataset = dataset.map(_augment, num_parallel_calls=min(N_THREADS, batch_size))
        return dataset

    def preprocess(self, dataset, batch_size=4):
        """
        data preprocessing here
        not much to do (assuming already preprocessed)
        but can crop to a smaller size, assuming images are already
            preregistered.
        """
        dataset = dataset.map(lambda x: _preprocess(x, self.cfg["crop_size"]),
                              num_parallel_calls=min(N_THREADS, batch_size))
        return dataset

    def generator_input_fn(self, file_list=None, batch_size=1, shuffle_buffer=-1, repeat=-1, augment=False, im_root="training"):

        generator = lambda : nifti_generator(
            file_list,
            os.path.join(self.cfg["nifti_path"], im_root),
            labels=self.labels,
            vol_labels=self.vol_labels,
            voxel_size_mm=self.cfg["rs_voxel_size"])

        ds = tf.data.Dataset.from_generator(
                generator,
                output_types=(
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )  #,
        # output_shapes=(tf.TensorShape([None, None, None]), tf.TensorShape([])))

        ds = ds.map(
            lambda f, v, e, m, s, l:
            {"image": f,
             "volume": tf.reshape(v, [self.feature_size["volume"]]),
             "entropy": tf.reshape(e, [self.feature_size["entropy"]]),
             "mean": tf.reshape(m, [self.feature_size["mean"]]),
             "stdev": tf.reshape(s, [self.feature_size["stdev"]]),
             "label": tf.reshape(l, [1])},
            num_parallel_calls=min(N_THREADS, batch_size)
        ).prefetch(self.cfg["max_concurrent_files"])

        if shuffle_buffer > 0:
            ds = ds.shuffle(shuffle_buffer)

        if repeat > 0:
            ds = ds.repeat()

        if augment:
            ds = self.augment(ds, batch_size)

        ds = self.preprocess(ds, batch_size)

        ds = ds.batch(batch_size).prefetch(1)

        batch = ds.make_one_shot_iterator().get_next()

        return {"image": batch["image"], "volume": batch["volume"],
                "entropy": batch["entropy"], "mean": batch["mean"],
                "stdev": batch["stdev"]}, batch["label"]

    def serving_input_fn(self):
        # in progress
        image = tf.placeholder(tf.float32, [None, None, None])
        volume = tf.placeholder(tf.float32, [self.feature_size["volume"]])
        entropy = tf.placeholder(tf.float32, [self.feature_size["entropy"]])

        features = {"image": image, "volume": volume, "entropy": entropy}
        features = _preprocess(features, self.cfg["crop_size"])
        features["image"] = tf.reshape(features["image"], [-1] + self.cfg["crop_size"] + [1])
        features["volume"] = tf.reshape(features["volume"], [-1, self.feature_size["volume"]])
        features["entropy"] = tf.reshape(features["entropy"], [-1, self.feature_size["entropy"]])
        receiving_input_tensors = {"image": image, "volume": volume, "entropy": entropy}

        return tf.estimator.export.ServingInputReceiver(features, receiving_input_tensors)

    def predict_input_fn(self, file_list=None):
        pass

    def write_tfrecords(self, output_dir):
        pass
