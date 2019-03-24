'''
Dual Path Networks
Combines ResNeXt grouped convolutions and DenseNet dense
connections to acheive state-of-the-art performance on ImageNet
References:
    - [Dual Path Networks](https://arxiv.org/abs/1707.01629)
    - https://github.com/titu1994/Keras-DualPathNetworks/blob/master/dual_path_network.py
    - https://github.com/wentaozhu/DeepLung/blob/master/detector/dpn3d26.py
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

__all__ = ['DualPathNetwork', 'DPN92', 'DPN98', 'DPN137', 'DPN107', 'preprocess_input', 'decode_predictions']


def _conv_bn(x, filters, kernel=(3, 3, 3), stride=(1, 1, 1), weight_decay=5e-4):
    ''' Adds a Batchnorm-Relu-Conv block for DPN
    Args:
        input: input tensor
        filters: number of output filters
        kernel: convolution kernel size
        stride: stride of convolution
    Returns: a keras tensor
    '''

    x = tf.keras.layers.Conv3D(
        filters, kernel, strides=stride,
        padding='same', use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

# doesn't work
def group_conv3d(x, filters, kernel_size, strides, groups=1, weight_decay=1e-4):
   assert filters % groups == 0, “number of filters is not divisible by groups”
   separate_filters = filters // groups
   grouped_x = tf.stack(tf.split(x, groups, axis=-1))  # groups, batch, x, y, z, channels

   grouped_x = tf.map_fn(
       lambda x: tf.keras.layers.Conv3D(
           separate_filters,
           kernel_size=kernel_size,
           strides=strides,
           use_bias=False,
           padding=“same”,
           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x),
       elems=grouped_x
   )
   x = tf.concat(tf.unstack(grouped_x), axis=-1)

   return x

# TODO : Working here
def _bottleneck(x, filters, n_dense, shortcut=False, kernel_size=(3, 3, 3), strides=(1, 1, 1), weight_decay=1e-4, is_training=False):
    if shortcut:
        sc = _conv_bn(x, filters+n_dense, kernel_size=(1, 1, 1), strides=strides)
    else:
        sc = x
    x = _conv_bn(x, filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), weight_decay=weight_decay, is_training=is_training)
    x = _conv_bn(x, filters, kernel_size=(3, 3, 3), strides=strides, weight_decay=weight_decay, is_training=is_training)
    x = _conv_bn(x, filters + n_dense, kernel_size=(1, 1, 1), strides=(1, 1, 1), weight_decay=weight_decay, is_training=is_training)

    x = tf.keras.layers.concatenate([sc[:, :, :, :, :filters] + x[:, :, :, :, :filters],
                                     sc[:, :, :, :, filters:],
                                     x[:, :, :, :, filters:]])
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    return x

def _dual_path_module(x, filters, n_dense, groups=8, shortcut=False, kernel_size=(3, 3, 3), strides=(1, 1, 1), weight_decay=1e-4, is_training=False):
    if shortcut:
        sc = _conv_bn(x, filters+n_dense, kernel_size=(1, 1, 1), strides=strides)
    else:
        sc = x
    bottleneck = _bottleneck(x, filters, n_dense,
                             groups=groups,
                             kernel_size=kernel_size,
                             strides=strides,
                             weight_decay=1e-4,
                             is_training=False)
    print(sc.shape)
    print(bottleneck.shape)
    out = tf.concat([
        bottleneck[:, :, :, :, :filters] + sc[:, :, :, :, :filters],
        sc[:, :, :, :, filters:],
        bottleneck[:, :, :, :, filters:]], axis=-1)
    return out


def _make_block(x, filters, n_dense, groups=8, n_bottlenecks=2, kernel_size=(3, 3, 3), strides=(1, 1, 1)):
    x = _dual_path_module(x,
                          filters=filters,
                          kernel_size=kernel_size,
                          groups=groups,
                          n_dense=n_dense,
                          shortcut=True,
                          strides=strides)
    for b in range(n_bottlenecks - 1):
        print(b)
        x = _dual_path_module(x, filters=filters, groups=groups, n_dense=n_dense, strides=(1, 1, 1))

    return x
#
#
# def _create_dpn(nb_classes, img_input, include_top, initial_conv_filters,
#                 filter_increment, depth, cardinality=32, width=3, weight_decay=5e-4, pooling=None):
#     ''' Creates a ResNeXt model with specified parameters
#     Args:
#         initial_conv_filters: number of features for the initial convolution
#         include_top: Flag to include the last dense layer
#         initial_conv_filters: number of features for the initial convolution
#         filter_increment: number of filters incremented per block, defined as a list.
#             DPN-92  = [16, 32, 24, 128]
#             DON-98  = [16, 32, 32, 128]
#             DPN-131 = [16, 32, 32, 128]
#             DPN-107 = [20, 64, 64, 128]
#         depth: number or layers in the each block, defined as a list.
#             DPN-92  = [3, 4, 20, 3]
#             DPN-98  = [3, 6, 20, 3]
#             DPN-131 = [4, 8, 28, 3]
#             DPN-107 = [4, 8, 20, 3]
#         width: width multiplier for network
#         weight_decay: weight_decay (tf.keras.regularizers.l2 norm)
#         pooling: Optional pooling mode for feature extraction
#             when `include_top` is `False`.
#             - `None` means that the output of the model will be
#                 the 4D tensor output of the
#                 last convolutional layer.
#             - `avg` means that global average pooling
#                 will be applied to the output of the
#                 last convolutional layer, and thus
#                 the output of the model will be a 2D tensor.
#             - `max` means that global max pooling will
#                 be applied.
#             - `max-avg` means that both global average and global max
#                 pooling will be applied to the output of the last
#                 convolution layer
#     Returns: a Keras Model
#     '''
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     N = list(depth)
#     base_filters = 256
#
#     # block 1 (initial conv block)
#     x = _initial_conv_block_inception(img_input, initial_conv_filters, weight_decay)
#
#     # block 2 (projection block)
#     filter_inc = filter_increment[0]
#     filters = int(cardinality * width)
#
#     x = _dual_path_block(x, pointwise_filters_a=filters,
#                          grouped_conv_filters_b=filters,
#                          pointwise_filters_c=base_filters,
#                          filter_increment=filter_inc,
#                          cardinality=cardinality,
#                          block_type='projection')
#
#     for i in range(N[0] - 1):
#         x = _dual_path_block(x, pointwise_filters_a=filters,
#                              grouped_conv_filters_b=filters,
#                              pointwise_filters_c=base_filters,
#                              filter_increment=filter_inc,
#                              cardinality=cardinality,
#                              block_type='normal')
#
#     # remaining blocks
#     for k in range(1, len(N)):
#         print("BLOCK %d" % (k + 1))
#         filter_inc = filter_increment[k]
#         filters *= 2
#         base_filters *= 2
#
#         x = _dual_path_block(x, pointwise_filters_a=filters,
#                              grouped_conv_filters_b=filters,
#                              pointwise_filters_c=base_filters,
#                              filter_increment=filter_inc,
#                              cardinality=cardinality,
#                              block_type='downsample')
#
#         for i in range(N[k] - 1):
#             x = _dual_path_block(x, pointwise_filters_a=filters,
#                                  grouped_conv_filters_b=filters,
#                                  pointwise_filters_c=base_filters,
#                                  filter_increment=filter_inc,
#                                  cardinality=cardinality,
#                                  block_type='normal')
#
#     x = concatenate(x, axis=channel_axis)
#
#     if include_top:
#         avg = GlobalAveragePooling2D()(x)
#         max = Globaltf.keras.layers.MaxPooling3D()(x)
#         x = add([avg, max])
#         x = tf.keras.layers.Lambda(lambda z: 0.5 * z)(x)
#         x = Dense(nb_classes, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
#                   kernel_initializer='he_normal', tf.keras.layers.Activation='softmax')(x)
#     else:
#         if pooling == 'avg':
#             x = GlobalAveragePooling2D()(x)
#         elif pooling == 'max':
#             x = Globaltf.keras.layers.MaxPooling3D()(x)
#         elif pooling == 'max-avg':
#             a = Globaltf.keras.layers.MaxPooling3D()(x)
#             b = GlobalAveragePooling2D()(x)
#             x = add([a, b])
#             x = tf.keras.layers.Lambda(lambda z: 0.5 * z)(x)
#
#     return x
#
# if __name__ == '__main__':
#     model = DPN92((224, 224, 3))
#     model.summary()
