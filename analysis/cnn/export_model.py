"""
Model exporting code
use to export a trained checkpoint to SavedModel for deployment
example call:
python stage4c_attr/export_model.py \
  --cfg=cfgs/checkpoint.json \
  --export_path="/tmp/test"
run the model using:
tensorflow_model_server \
    --port=9000 \
    --model_base_path=/tmp/test \
    --model_name=attribute
or can use deploy.py
@maintainer: jeffc@voxelcloud.io
"""
import os
import sys
import json
import tensorflow as tf

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

tf.app.flags.DEFINE_string('cfg', '', 'cfg file')
tf.app.flags.DEFINE_string('export_path', '/tmp/test_export', 'export path')
FLAGS = tf.app.flags.FLAGS


def main(_):
    with open(FLAGS.cfg, "rt") as config_file:
        cfg = json.load(config_file)

    tf.logging.info("Building model")
    model_fn, params = create_estimator_from_subclass(_MODELS[cfg["model"]], cfg)
    tf.logging.info("done")
    dl = DataLoader()

    session_config = tf.ConfigProto(
       inter_op_parallelism_threads=2,  # per https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
       intra_op_parallelism_threads=MULTI_THREAD
    )

    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
            model_dir=os.path.join(cfg["save_dir"]),
            save_summary_steps=cfg["report_every"],
            save_checkpoints_steps=cfg["save_every"],
            keep_checkpoint_max=100,
            session_config=session_config
    )

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        warm_start_from=None
    )

    classifier.export_saved_model(
        export_dir_base=FLAGS.export_path,
        serving_input_receiver_fn=dl.serving_input_fn,
        assets_extra=None,
        as_text=False,
        checkpoint_path=cfg["checkpoint_path"])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main=main)
