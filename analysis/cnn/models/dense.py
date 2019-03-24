import tensorflow as tf


from .layers import _conv_leaky, _conv_bn_leaky


class DenseModel(tf.keras.Model):
    def __init__(self, dropout_rate=0.2, weight_decay=1e-6, **unused):
        super(DenseModel, self).__init__()
        self.concat_1 = tf.keras.layers.Concatenate()
        self.drop_1 = tf.keras.layers.Dropout(rate=dropout_rate)  # assume noisy labels
        self.dense_1 = tf.keras.layers.Dense(128, activation = tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.dense_2 = tf.keras.layers.Dense(64, activation = tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.dense_3 = tf.keras.layers.Dense(32, activation = tf.nn.relu , kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.logits = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.concat_1([inputs["volume"], inputs["entropy"], inputs["mean"], inputs["stdev"]])
        x = self.drop_1(x, training=training)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.logits(x)


def dense_net(inputs, is_training=False, weight_decay=1e-4):
    dtype=tf.float32
    end_points = {}

    with tf.variable_scope("features"):
        feat_net = tf.keras.layers.Concatenate()([inputs["volume"], inputs["entropy"]])
        end_points["dense_1"] = feat_net = tf.keras.layers.Dense(64, activation = tf.nn.relu)(feat_net)

    with tf.variable_scope("readout"):
        end_points["dense_2"] = net = tf.keras.layers.Dense(128, activation = tf.nn.relu)(feat_net)

    with tf.variable_scope("logits"):
        end_points["logits"] = logits = tf.keras.layers.Dense(1)(net)

    l2_loss = tf.losses.get_regularization_loss() if weight_decay else None

    return logits, end_points, l2_loss
    # return inputs, logits
