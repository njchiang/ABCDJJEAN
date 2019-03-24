import tensorflow as tf

def _cast_layer(datatype):
    return tf.keras.layers.Lambda(lambda x: tf.cast(x, datatype))

class ConvBNLeakyBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, alpha=0.3, conv_args={}, bn_args={}):
        super(ConvBNLeakyBlock, self).__init__()

        self.conv = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, use_bias=False, **conv_args)
        self.bn = tf.keras.layers.BatchNormalization(**bn_args)
        self.leaky = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = self.leaky(x)

        return x


def _conv_bn_leaky(x, filters, kernel_size, is_training=False, alpha=0.3, dtype=tf.float32, conv_args={}, bn_args={}):
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               **conv_args)(x)
    # x = _cast_layer(tf.float32)(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x, training=is_training)
    # x = _cast_layer(dtype)(x)

    return tf.keras.layers.LeakyReLU(alpha=alpha)(x)

def _conv_leaky(x, filters, kernel_size, alpha=0.3, conv_args={}):
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               **conv_args)(x)
    # x = _cast_layer(tf.float32)(x)
    # x = _cast_layer(tf.float16)(x)

    return tf.keras.layers.LeakyReLU(alpha=alpha)(x)

def _projection_shortcut(x, filters):
    return tf.keras.layers.Conv3D(filters=filters, kernel_size=[1,1,1])(x)

def _cast_layer(datatype):
    return tf.keras.layers.Lambda(lambda x: tf.cast(x, datatype))
