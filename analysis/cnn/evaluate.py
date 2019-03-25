from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string("predictions", "", "path to config file")
tf.app.flags.DEFINE_string("labels", "D:\\fmri\\ABCD\\data\\results\\gt_validation.csv", "path to labels file")

FLAGS = tf.app.flags.FLAGS

def main(_):
    labels = pd.read_csv(FLAGS.labels).set_index("subject")
    preds = pd.read_csv(FLAGS.predictions).set_index("subject")
    df = labels.join(preds, on="subject").fillna(0)  # check this:

    df["SSE"] = (df["predicted_score"] - df["fluid.resid"]) ** 2

    df.to_csv("{}-results.csv".format(os.path.splitext(FLAGS.predictions)[0]))
    print(df.mean())
    df.plot(x="fluid.resid", y="predicted_score", kind="scatter")
    plt.show()
    
""" testing the code """
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)

    tf.app.run(main=main)
