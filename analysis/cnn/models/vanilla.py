import tensorflow as tf


from .layers import _conv_leaky, _conv_bn_leaky, ConvBNLeakyBlock

class VanillaModel(tf.keras.Model):
    def __init__(self, weight_decay=1e-4, dtype=tf.float32, **unused):
        super(VanillaModel, self).__init__()
        self.initial_conv = ConvBNLeakyBlock(16, [15, 15, 15], conv_args={"padding": "valid", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
        self.conv_block_1 = ConvBNLeakyBlock(16, [3, 3, 3], conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
        self.pool_1 = tf.keras.layers.MaxPool3D(pool_size=[2,2,2], strides=[2,2,2])
        self.conv_block_2 = ConvBNLeakyBlock(32, [3, 3, 3], conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
        self.pool_2 = tf.keras.layers.MaxPool3D(pool_size=[2,2,2], strides=[2,2,2])
        self.conv_block_3 = ConvBNLeakyBlock(64, [3, 3, 3], conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
        self.pool_3 = tf.keras.layers.MaxPool3D(pool_size=[2,2,2], strides=[2,2,2])
        self.conv_block_4 = ConvBNLeakyBlock(64, [3, 3, 3], conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
        self.pool_4 = tf.keras.layers.MaxPool3D(pool_size=[2,2,2], strides=[2,2,2])
        self.conv_block_5 = ConvBNLeakyBlock(64, [3, 3, 3], conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
        self.pool_5 = tf.keras.layers.GlobalAveragePooling3D()

        self.concat_1 = tf.keras.layers.Concatenate()
        self.drop_1 = tf.keras.layers.Dropout(0.2)
        self.dense_1 = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))  # this wasn't working for some reason
        self.dense_2 = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))  # this wasn't working for some reason

        self.merge = tf.keras.layers.Add()

        self.dense_3 =  tf.keras.layers.Dense(128)  #, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))  # this wasn't working for some reason
        self.logits = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        im_net = self.initial_conv(inputs["image"])
        im_net = self.conv_block_1(im_net, training=training)
        im_net = self.pool_1(im_net)
        im_net = self.conv_block_2(im_net, training=training)
        im_net = self.pool_2(im_net)
        im_net = self.conv_block_3(im_net, training=training)
        im_net = self.pool_3(im_net)
        im_net = self.conv_block_4(im_net, training=training)
        im_net = self.pool_4(im_net)
        im_net = self.conv_block_5(im_net, training=training)
        im_net = self.pool_5(im_net)

        feat_net = self.concat_1([inputs["volume"], inputs["entropy"]])
        feat_net = self.drop_1(feat_net, training=training)
        feat_net = tf.nn.relu(self.dense_1(feat_net))
        feat_net = tf.nn.relu(self.dense_2(feat_net))

        net = self.merge([im_net, feat_net])
        net = tf.nn.relu(self.dense_3(net))

        return self.logits(net)


def vanilla_cnn(inputs, is_training=False, weight_decay=1e-4):
    dtype=tf.float32
    end_points = {}
    with tf.variable_scope("model"):
        with tf.variable_scope("image"):
            end_points["initial_conv"] = im_net = _conv_bn_leaky(inputs["image"], 16, [15, 15, 15], is_training=is_training, dtype=dtype, conv_args={"padding": "valid", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
            end_points["conv_1"] = im_net = _conv_bn_leaky(im_net, 16, [3, 3, 3], is_training=is_training, dtype=dtype, conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
            end_points["pool_1"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["conv_2"] = im_net = _conv_bn_leaky(im_net, 32, [3, 3, 3], is_training=is_training, dtype=dtype, conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
            end_points["pool_2"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["conv_3"] = im_net = _conv_bn_leaky(im_net, 64, [3, 3, 3], is_training=is_training, dtype=dtype, conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
            end_points["pool_3"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["conv_4"] = im_net = _conv_bn_leaky(im_net, 64, [3, 3, 3], is_training=is_training, dtype=dtype, conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
            end_points["pool_4"] = im_net = tf.keras.layers.MaxPool3D(pool_size=[2,2,2],
                                            strides=[2,2,2])(im_net)

            end_points["conv_5"] = im_net = _conv_bn_leaky(im_net, 64, [3, 3, 3], is_training=is_training, dtype=dtype, conv_args={"padding": "same", "kernel_regularizer": tf.keras.regularizers.l2(weight_decay)})
            end_points["pool_5"] = im_net = tf.keras.layers.GlobalAveragePooling3D()(im_net)

    with tf.variable_scope("features"):
        feat_net = tf.keras.layers.Concatenate()([inputs["volume"], inputs["entropy"]])
        end_points["dense_1"] = feat_net = tf.keras.layers.Dense(64, activation = tf.nn.relu)(feat_net)

    with tf.variable_scope("readout"):
        end_points["merge"] = net = tf.keras.layers.Add()([im_net, feat_net])
        end_points["dense_2"] = net = tf.keras.layers.Dense(128, activation = tf.nn.relu)(net)

    with tf.variable_scope("logits"):
        end_points["logits"] = logits = tf.keras.layers.Dense(1)(net)

    l2_loss = tf.losses.get_regularization_loss() if weight_decay else None

    return logits, end_points, l2_loss
    # return inputs, logits
