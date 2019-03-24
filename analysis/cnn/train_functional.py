"""
Model function generator

main author: Jeff
"""
import os
import json
import tensorflow as tf

from models import create_estimator_from_functional
from models.resnet import residual_cnn
from models.vanilla import vanilla_cnn
from models.dense import dense_net
from data import DataLoader

_MODELS = {
    "dense": dense_net,
    "vanilla": vanilla_cnn,
    "resnet": residual_cnn
}

tf.app.flags.DEFINE_string("cfg", "cfgs/dev.json", "path to config file")
tf.app.flags.DEFINE_string("gpu", "0", "gpu id")

MULTI_THREAD = 4

def main(_):
    # setup config dictionary
    with open(FLAGS.cfg, "r") as f:
        config = json.load(f)

    session_config = tf.ConfigProto(
       inter_op_parallelism_threads=2,  # per https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
       intra_op_parallelism_threads=MULTI_THREAD
    )

    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
       model_dir=os.path.join(config["save_dir"]),
       save_summary_steps=config["report_every"],
       save_checkpoints_steps=config["save_every"],
       keep_checkpoint_max=config["max_num_saved_models"],
       session_config=session_config
    )
    exclude_query = ".*"
    if config["checkpoint_path"] != "":
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=config["checkpoint_path"],
            vars_to_warm_start=exclude_query
        )
    elif config["pretrained_model"] != "":
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=config["pretrained_model"],
            vars_to_warm_start=exclude_query
        )
    else:
        ws = None

    dl = DataLoader()
    # model_fn, params = create_estimator_model(residual_cnn, config)
    # model_fn, params = create_estimator_model(vanilla_cnn, config)
    model_fn, params = create_estimator_from_functional(_MODELS[config["model"]], config)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=None,
        config=run_config,
        warm_start_from=ws
    )

    if config["train_path"] is not None and \
       config["inference_path"] is not None:

        tf.logging.info("Training and evaluating...")
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: dl.generator_input_fn(
                file_list=config["train_path"],
                batch_size=config["batch_size"],
                augment=True,
                shuffle_buffer=4 * config["batch_size"],
                repeat=1
            ),
            max_steps=config["max_steps"],
            # hooks=None  # deal with this later. TODO
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: dl.generator_input_fn(
                file_list=config["inference_path"],
                batch_size=config["batch_size"],
                augment=False,
                shuffle_buffer=-1,
                repeat=-1,
                im_root="val"
            ),
            steps=None,  # evaluate on entire set
            name="validation",  # can have multiple validation runs, each will be in different directory
            exporters=None,  # Figure this out later
            # hooks=None,
            start_delay_secs=120,  # as_default
            throttle_secs=0  # default
        )

        tf.estimator.train_and_evaluate(
            estimator=classifier,
            train_spec=train_spec,
            eval_spec=eval_spec
        )

    else:
        tf.logging.info(
            "No inference path: training for {} steps over {} epochs".format(
                config["max_steps"], config["epochs"]
            )
        )

        classifier = classifier.train(
            input_fn=lambda: dl.generator_input_fn(
                file_list=config["train_path"],
                batch_size=config["batch_size"],
                augment=True,
                preprocess=True,
                shuffle_buffer=1000,
                repeat=1
            ),
            hooks=None,  # TODO
            steps=config["max_steps"],
            max_steps=None,  # leave this as none
            saving_listeners=None  # TODO
         )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = tf.app.flags.FLAGS
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.app.run()
