from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import json
import pandas as pd

from itertools import chain, islice


from models import create_estimator_from_subclass
# from models.resnet import residual_cnn
from models.vanilla import VanillaModel
from models.dense import DenseModel
from data import DataLoader, nifti_generator

_MODELS = {
    "dense": DenseModel,
    "vanilla": VanillaModel
    # "resnet": residual_cnn
}

MULTI_THREAD = 4
""" useful parameters for when testing """

tf.app.flags.DEFINE_string("cfg", "cfgs/dev.json", "path to config file")
tf.app.flags.DEFINE_string("guids", "D:\\fmri\\ABCD\\data\\jeff_val_guids.txt", "path to config file")
tf.app.flags.DEFINE_string("mode", "val", "evaluation set")
tf.app.flags.DEFINE_string("gpu", "", "gpu id")

tf.app.flags.DEFINE_string('output_dir', 'D:\\fmri\\ABCD\\outputs', 'path to output')

FLAGS = tf.app.flags.FLAGS


def batch_fn(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def init_tf_serving(cfg):
    """
    Creates the model (tf.Estimator)
    inputs:
    cfg: dict of configuration parameters
    returns:
    data_handler: data handler object
    classifier: tf.Estimator attribute model
    params: tf.Estimator parameters
    """
    model_fn, params = create_estimator_from_subclass(_MODELS[cfg["model"]], cfg)

    session_config = tf.ConfigProto(
       inter_op_parallelism_threads=2,  # per https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
       intra_op_parallelism_threads=MULTI_THREAD
    )
    run_config = tf.estimator.RunConfig(
       model_dir=os.path.join(cfg["save_dir"]),
       save_summary_steps=cfg["report_every"],
       save_checkpoints_steps=cfg["save_every"],
       keep_checkpoint_max=cfg["max_num_saved_models"],
       session_config=session_config
    )
    session_config.gpu_options.allow_growth = True

    assert "checkpoint_path" in cfg, "No path to checkpoint specified in config"

    tf.logging.info("Building model")
    model_fn, params = create_estimator_from_subclass(_MODELS[cfg["model"]], cfg)
    tf.logging.info("done")
    data_handler = DataLoader()

    # to allow for compatibility with tf 1.8
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=None,
        config=run_config)

    params["batch_size"] = cfg["batch_size"]
    params["checkpoint_path"] = cfg["checkpoint_path"]
    return data_handler, classifier, params


def process_serving(guid_file, data_handler, classifier, params):
    """
    Runs inference using tf.Estimator.predict()
    inputs:
    returns:
      dets: detections dictionary with characteristics appended
    """

    # ct = (ct - ct.min()) / (ct.max() - ct.min())  # normalize to 0-1, should already be done

    # ugly- zip this with nifti generator...
    with open(guid_file, "r") as f:
        subjects = [s.strip() for s in f.readlines()]

    predictions = classifier.predict(
        input_fn = lambda: data_handler.generator_input_fn(
            file_list=guid_file,
            batch_size=params["batch_size"],
            augment=False,
            shuffle_buffer=-1,
            repeat=-1,
            im_root=FLAGS.mode),
        checkpoint_path=params["checkpoint_path"]
    )

    results = {"subject": [], "predicted_score": []}

    for pred in predictions:
        results["predicted_score"].append(pred["Predicted-residual"].squeeze())
        results["subject"].append(pred["subject"].decode())
        print("{}: {}".format(pred["subject"].decode(), pred["Predicted-residual"]))

    # for sub, pred in zip(subjects, predictions):
    #     results["predicted_score"].append(pred["Predicted-residual"].squeeze())
    #     results["subject"].append(pred["subject"])
    #     print("{}: {}".format(sub, pred))
    return results


def main(_):
    tf.logging.info("Imported modules")

    if FLAGS.mode not in ["training", "val", "testing"]:
        raise ValueError("mode must be training/val/testing")

    with open(FLAGS.cfg, "r") as f:
        cfg = json.load(f)
    tf.logging.info("JSON loaded")

    data_handler, classifier, params = init_tf_serving(cfg)
    tf.logging.info("Server initialized")

    results = process_serving(FLAGS.guids, data_handler, classifier, params)
    tf.logging.info("Processed")
    out_file_name = "{}-{}.csv".format(os.path.basename(params["checkpoint_path"]), FLAGS.mode)
    pd.DataFrame(results).set_index("subject").to_csv(os.path.join(FLAGS.output_dir, out_file_name))


""" testing the code """
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.logging.info("Processing on GPU: {}".format(FLAGS.gpu))
    tf.app.run(main=main)
