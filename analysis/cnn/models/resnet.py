"""
Tensorflow Model library

Using tf.keras whenever possible
too large of a model...

Main author: Jeff
"""

import tensorflow as tf

from .layers import _conv_bn_leaky, _conv_leaky, _projection_shortcut

def fp16_loss_generator(loss_fn, multiplier=1):
    def custom_loss(yTrue, yPred):
        return multiplier * loss_fn(yTrue, yPred)
    return custom_loss

def resblock(x, filters, depth=2, is_training=False, kernel=[3, 3, 3], weight_decay=1e-4, dtype=tf.float32):
    residual = _projection_shortcut(x, filters)
    for _ in range(depth):
        x = _conv_bn_leaky(x, filters, kernel, is_training=is_training, dtype=dtype, conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
    x = tf.keras.layers.Add()([residual, x])
    return x

def residual_cnn(inputs, is_training=False, weight_decay=1e-4):
    end_points = {}
    with tf.variable_scope("model"):
        with tf.variable_scope("image"):
            im_net = tf.keras.layers.Conv3D(filters=8,
                                         kernel_size=[7, 7, 7],
                                         padding="valid",
                                         activation=None)(inputs["image"])

            # net = tf.keras.layers.BatchNormalization()(net)
            end_points["initial_conv"] = im_net = tf.keras.layers.LeakyReLU()(im_net)

            end_points["resblock_1"] = im_net = resblock(im_net, 8, is_training=is_training)
            end_points["maxpool_1"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["resblock_2"] = im_net = resblock(im_net, 16, is_training=is_training)
            end_points["maxpool_2"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["resblock_3"] = im_net = resblock(im_net, 32, is_training=is_training)
            end_points["maxpool_3"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["resblock_4"] = im_net = resblock(im_net, 64, is_training=is_training)
            end_points["maxpool_4"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["resblock_5"] = im_net = resblock(im_net, 128, is_training=is_training)

            end_points["global_pool"] = im_net = tf.keras.layers.GlobalAveragePooling3D()(im_net)

        with tf.variable_scope("features"):
            feat_net = tf.keras.layers.Concatenate()([inputs["volume"], inputs["entropy"]])
            end_points["dense_1"] = feat_net = tf.keras.layers.Dense(128, activation = tf.nn.relu)(feat_net)

        with tf.variable_scope("readout"):
            end_points["merge"] = net = tf.keras.layers.Add()([im_net, feat_net])
            end_points["dense_2"] = net = tf.keras.layers.Dense(256, activation = tf.nn.relu)(net)

        with tf.variable_scope("logits"):
            end_points["logits"] = logits = tf.keras.layers.Dense(1)(net)

    l2_loss = tf.losses.get_regularization_loss() if weight_decay else None

    return logits, end_points, l2_loss
    # return inputs, logits


def memtest_cnn(inputs):
    net = tf.keras.layers.Conv3D(filters=32,
                                 kernel_size=[15, 15, 15],
                                 padding="valid",
                                 activation=None)(inputs)
    # net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)

    # residual = _projection_shortcut(net, 32)

    net = _conv_leaky(net, 32, [7, 7, 7], conv_args={"padding": "same"})
    # net = _conv_leaky(net, 32, [7, 7, 7], conv_args={"padding": "same"})
    # net = tf.keras.layers.Add()([residual, net])
    net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                    strides=[2,2,2])(net)

    # residual = _projection_shortcut(net, 64)

    net = _conv_leaky(net, 64, [5, 5, 5], conv_args={"padding": "same"})
    # net = _conv_leaky(net, 64, [5, 5, 5], conv_args={"padding": "same"})
    # net = tf.keras.layers.Add()([residual, net])
    net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                    strides=[2,2,2])(net)

    # residual = _projection_shortcut(net, 128)
    net = _conv_leaky(net, 128, [5, 5, 5], conv_args={"padding": "same"})
    # net = _conv_leaky(net, 128, [5, 5, 5], conv_args={"padding": "same"})
    # net = tf.keras.layers.Add()([residual, net])
    net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                    strides=[2,2,2])(net)

    # residual = _projection_shortcut(net, 256)
    net = _conv_leaky(net, 256, [5, 5, 5], conv_args={"padding": "same"})
    # net = _conv_leaky(net, 256, [5, 5, 5], conv_args={"padding": "same"})
    # net = tf.keras.layers.Add()([residual, net])
    net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                    strides=[2,2,2])(net)

    # residual = _projection_shortcut(net, 256)
    net = _conv_leaky(net, 256, [5, 5, 5], conv_args={"padding": "same"})
    # net = _conv_leaky(net, 256, [5, 5, 5], conv_args={"padding": "same"})
    # net = tf.keras.layers.Add()([residual, net])

    net = tf.keras.layers.GlobalAveragePooling3D()(net)

    net = tf.keras.layers.Dense(32, activation = tf.nn.leaky_relu)(net)

    logits = tf.keras.layers.Dense(1)(net)

    return logits
