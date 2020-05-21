import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

X = x_train
'''
print(X.shape)
#see some images
for _ in range(10):
    i = np.random.randint(1,50000)
    plt.imshow(X[i])
'''


#list of hyperparameters
BATCHSIZE = 32
LEARNING_RATE = 0.0002
TRAINING_RATIO = 1
BETA_1 = 0.0
BETA_2 = 0.9
EPOCHS = 500
BN_MOMENTUM = 0.1
BN_EPSILON = 0.00002

GENERATE_ROW_NUM = 8
GENERATE_BATCHSIZE = GENERATE_ROW_NUM * GENERATE_ROW_NUM

''' Implementation of Spectral Normalization'''
from keras import backend as K
from keras.engine import *
from keras.legacy import interfaces
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers import Dense, Conv1D, Conv2D, Conv3D, Conv2DTranspose, Embedding
import tensorflow as tf

class DenseSN(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

class _ConvSN(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 **kwargs):
        super(_ConvSN, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.spectral_normalization = spectral_normalization
        self.u = None

    def _l2normalize(self, v, eps=1e-12):
        return v / (K.sum(v ** 2) ** 0.5 + eps)

    def power_iteration(self, u, W):
        '''
        Accroding the paper, we only need to do power iteration one time.
        '''
        v = self._l2normalize(K.dot(u, K.transpose(W)))
        u = self._l2normalize(K.dot(v, W))
        return u, v
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        #Spectral Normalization
        if self.spectral_normalization:
            self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                     initializer=initializers.RandomNormal(0, 1),
                                     name='sn',
                                     trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        if self.spectral_normalization:
            W_shape = self.kernel.shape.as_list()
            #Flatten the Tensor
            W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
            _u, _v = power_iteration(W_reshaped, self.u)
            #Calculate Sigma
            sigma=K.dot(_v, W_reshaped)
            sigma=K.dot(sigma, K.transpose(_u))
            #normalize it
            W_bar = W_reshaped / sigma
            #reshape weight tensor
            if training in {0, False}:
                W_bar = K.reshape(W_bar, W_shape)
            else:
                with tf.control_dependencies([self.u.assign(_u)]):
                    W_bar = K.reshape(W_bar, W_shape)

            #update weitht
            self.kernel = W_bar

        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvSN2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class ConvSN1D(Conv1D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                 initializer=initializers.RandomNormal(0, 1),
                 name='sn',
                 trainable=False)
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv1d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class ConvSN3D(Conv3D):
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv3d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class EmbeddingSN(Embedding):

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)

        self.u = self.add_weight(shape=tuple([1, self.embeddings.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

        self.built = True

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.embeddings.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.embeddings, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        self.embeddings = W_bar

        out = K.gather(self.embeddings, inputs)
        return out

class ConvSN2DTranspose(Conv2DTranspose):

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height,
                                              stride_h, kernel_h,
                                              self.padding,
                                              out_pad_h)
        out_width = conv_utils.deconv_length(width,
                                             stride_w, kernel_w,
                                             self.padding,
                                             out_pad_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        #Spectral Normalization
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        self.kernel = W_bar

        outputs = K.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

''' End of spectral normalization implementation'''


'''Implementation of attention layers '''

from keras.layers import Layer
class attention_generator(Layer):
  def __init__(self,**kwargs):
    channels = 64
    self.channels = channels
    self.filters_f = self.channels//8
    self.filters_g = self.channels//8
    self.filters_h = self.channels
    super(attention_generator,self).__init__(**kwargs)

  def build(self,input_shape):
    kernel_shape_f = (1,1) + (self.channels,self.filters_f)
    kernel_shape_g = (1,1) + (self.channels,self.filters_g)
    kernel_shape_h = (1,1) + (self.channels,self.filters_h)

    self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

    self.kernel_f = self.add_weight(shape = kernel_shape_f,initializer = 'glorot_uniform',name = 'kernel_f')
    self.kernel_g = self.add_weight(shape = kernel_shape_g,initializer = 'glorot_uniform',name = 'kernel_g')
    self.kernel_h = self.add_weight(shape = kernel_shape_h,initializer = 'glorot_uniform',name = 'kernel_g')

    super(attention_generator,self).build(input_shape)
    self.input_spec = tf.keras.layers.InputSpec(ndim=4, axes={3: input_shape[-1]})
    self.built = True

  def call(self,x):
    def hw_flatten(x) :
      return K.reshape(x,[K.shape(x)[0], -1, K.shape(x)[-1]])
    f = K.conv2d(x,
                  kernel=self.kernel_f,
                  strides=(1, 1), padding='same')  # [bs, h, w, c']

    g = K.conv2d(x,
                  kernel=self.kernel_g,
                  strides=(1, 1), padding='same')  # [bs, h, w, c']

    h = K.conv2d(x,
                  kernel=self.kernel_h,
                  strides=(1, 1), padding='same')  # [bs, h, w, c]


    val = K.shape(hw_flatten(f))


    s = K.batch_dot(hw_flatten(g),K.reshape(hw_flatten(f),[val[0],val[2],val[1]]))  # # [bs, N, N]

    beta = K.softmax(s, axis=-1)  # attention map

    o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
    o = K.reshape(o,K.shape(x))  # [bs, h, w, C]
    x = self.gamma * o + x
    return x


  def compute_output_shape(self, input_shape):
    return input_shape

from keras.layers import Layer
class attention_discriminator(Layer):
  def __init__(self,**kwargs):
    channels = 32
    self.channels = channels
    self.filters_f = self.channels//8
    self.filters_g = self.channels//8
    self.filters_h = self.channels
    super(attention_discriminator,self).__init__(**kwargs)

  def build(self,input_shape):
    kernel_shape_f = (1,1) + (self.channels,self.filters_f)
    kernel_shape_g = (1,1) + (self.channels,self.filters_g)
    kernel_shape_h = (1,1) + (self.channels,self.filters_h)

    self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

    self.kernel_f = self.add_weight(shape = kernel_shape_f,initializer = 'glorot_uniform',name = 'kernel_f')
    self.kernel_g = self.add_weight(shape = kernel_shape_g,initializer = 'glorot_uniform',name = 'kernel_g')
    self.kernel_h = self.add_weight(shape = kernel_shape_h,initializer = 'glorot_uniform',name = 'kernel_g')

    super(attention_discriminator,self).build(input_shape)
    self.input_spec = tf.keras.layers.InputSpec(ndim=4, axes={3: input_shape[-1]})
    self.built = True

  def call(self,x):
    def hw_flatten(x) :
      return K.reshape(x,[K.shape(x)[0], -1, K.shape(x)[-1]])
    f = K.conv2d(x,
                  kernel=self.kernel_f,
                  strides=(1, 1), padding='same')  # [bs, h, w, c']

    g = K.conv2d(x,
                  kernel=self.kernel_g,
                  strides=(1, 1), padding='same')  # [bs, h, w, c']

    h = K.conv2d(x,
                  kernel=self.kernel_h,
                  strides=(1, 1), padding='same')  # [bs, h, w, c]


    val = K.shape(hw_flatten(f))


    s = K.batch_dot(hw_flatten(g),K.reshape(hw_flatten(f),[val[0],val[2],val[1]]))  # # [bs, N, N]

    beta = K.softmax(s, axis=-1)  # attention map

    o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
    o = K.reshape(o,K.shape(x))  # [bs, h, w, C]
    x = self.gamma * o + x
    return x


  def compute_output_shape(self, input_shape):
    return input_shape

''' End of implementation of attention layers'''



''' To calculate FID score'''

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

# scale the list of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		new_image = resize(image, new_shape, 0)
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	covmean = sqrtm(sigma1.dot(sigma2))
	if iscomplexobj(covmean):
		covmean = covmean.real
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))






'''
#DCGAN final model trained for 400 epochs.

DCGAN_generator_400  = load_model('DCGAN_generator_400.h5',custom_objects = {'DenseSN':DenseSN,'ConvSN2D':ConvSN2D,'ConvSN2DTranspose':ConvSN2DTranspose})
test_noise = np.random.randn(64,32)
generated_image = DCGAN_generator_400.predict(test_noise)
generator_image = ((generated_image + 1)/ 2)
for i in range(GENERATE_ROW_NUM):
  new = generated_image[i * GENERATE_ROW_NUM : i * GENERATE_ROW_NUM + GENERATE_ROW_NUM].reshape(32 *GENERATE_ROW_NUM,32,3)
  if i != 0:
    old = np.concatenate((old,new),axis = 1)
  else:
    old = new

# plotting the 8 X 8 grid of images
plt.figure()
fig = plt.imshow(old)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

#generating some images and viewing the same
some_images = np.random.randn(10,32)
some_images = DCGAN_generator_400.predict(some_images)
some_images = [((image + 1)/2) for image in some_images]


for image in some_images:
  plt.figure(figsize = (1.3,1.3))
  fig = plt.imshow(image)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)


val_images = x_test
shuffle(val_images)
val_images = val_images[:1000]
val_images = scale_images(val_images,(299,299,3))
final_images = np.random.randn(1000,32)
final_images = DCGAN_generator_400.predict(final_images)
final_images = [(image + 1)/2 for image in final_images]
final_images = scale_images(final_images,(299,299,3))
fid_score = calculate_fid(model,val_images,final_images)
print("FID_score is ",fid_score)
'''




'''
# SAGAN final model trained for 300 epochs

SAGAN_generator_300  = load_model('SAGAN_generator_300.h5',custom_objects = {'DenseSN':DenseSN,'ConvSN2D':ConvSN2D,'ConvSN2DTranspose':ConvSN2DTranspose,'attention_generator':attention_generator,'attention_discriminator':attention_discriminator})
test_noise = np.random.randn(64,32)
generated_image = SAGAN_generator_300.predict(test_noise)
generator_image = ((generated_image + 1)/ 2)
for i in range(GENERATE_ROW_NUM):
  new = generated_image[i * GENERATE_ROW_NUM : i * GENERATE_ROW_NUM + GENERATE_ROW_NUM].reshape(32 *GENERATE_ROW_NUM,32,3)
  if i != 0:
    old = np.concatenate((old,new),axis = 1)
  else:
    old = new

#plotting the 8 X 8 grid ofimages
plt.figure()
fig = plt.imshow(old)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

#Plotting some images generated by the model and viewing the same
some_images = np.random.randn(10,32)
some_images = SAGAN_generator_300.predict(some_images)
some_images = [((image + 1)/2) for image in some_images]


for image in some_images:
  plt.figure(figsize = (1.3,1.3))
  fig = plt.imshow(image)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)


val_images = x_test
shuffle(val_images)
val_images = val_images[:1000]
val_images = scale_images(val_images,(299,299,3))
final_images = np.random.randn(1000,32)
final_images = SAGAN_generator_300.predict(final_images)
final_images = [(image + 1)/2 for image in final_images]
final_images = scale_images(final_images,(299,299,3))
fid_score = calculate_fid(model,val_images,final_images)
print("FID_score is ",fid_score)
'''

''' DCGAN final model '''

DCGAN_GENERATOR  = load_model('DCGAN_GENERATOR.h5',custom_objects = {'DenseSN':DenseSN,'ConvSN2D':ConvSN2D,'ConvSN2DTranspose':ConvSN2DTranspose})
test_noise = np.random.randn(64,32)
generated_image = DCGAN_GENERATOR.predict(test_noise)
generator_image = ((generated_image + 1)/ 2)
for i in range(GENERATE_ROW_NUM):
  new = generated_image[i * GENERATE_ROW_NUM : i * GENERATE_ROW_NUM + GENERATE_ROW_NUM].reshape(32 *GENERATE_ROW_NUM,32,3)
  if i != 0:
    old = np.concatenate((old,new),axis = 1)
  else:
    old = new

#Plotting the 8 X 8 grid of iamges generated by DCGAN
print("DCGAN")
plt.figure()
fig = plt.imshow(old)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

'''
# plot some of the images generated by the model
some_images = np.random.randn(10 * 2,32)
some_images = DCGAN_GENERATOR.predict(some_images)
some_images = [((image + 1)/2) for image in some_images]


for image in some_images:
  plt.figure(figsize = (1.3,1.3))
  fig = plt.imshow(image)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
'''

val_images = x_test
shuffle(val_images)
val_images = val_images[:1000]
val_images = scale_images(val_images,(299,299,3))
final_images = np.random.randn(1000,32)
final_images = DCGAN_GENERATOR.predict(final_images)
final_images = [(image + 1)/2 for image in final_images]
final_images = scale_images(final_images,(299,299,3))
fid_score = calculate_fid(model,val_images,final_images)
print("FID score of DCGAN is ",fid_score)



''' SAGAN final model '''

SAGAN_GENERATOR  = load_model('SAGAN_GENERATOR.h5',custom_objects = {'DenseSN':DenseSN,'ConvSN2D':ConvSN2D,'ConvSN2DTranspose':ConvSN2DTranspose,'attention_generator':attention_generator,'attention_discriminator':attention_discriminator})
test_noise = np.random.randn(64,32)
generated_image = SAGAN_GENERATOR.predict(test_noise)
generator_image = ((generated_image + 1)/ 2)
for i in range(GENERATE_ROW_NUM):
  new = generated_image[i * GENERATE_ROW_NUM : i * GENERATE_ROW_NUM + GENERATE_ROW_NUM].reshape(32 *GENERATE_ROW_NUM,32,3)
  if i != 0:
    old = np.concatenate((old,new),axis = 1)
  else:
    old = new

#plotting 8 X 8 grid of images generated by SAGAN.
print("SAGAN")
plt.figure()
fig = plt.imshow(old)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
'''
#plotting some of the images generated by the generator.
some_images = np.random.randn(20,32)
some_images = SAGAN_GENERATOR.predict(some_images)
some_images = [((image + 1)/2) for image in some_images]


for image in some_images:
  plt.figure(figsize = (1.3,1.3))
  fig = plt.imshow(image)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
'''

val_images = x_test
shuffle(val_images)
val_images = val_images[:1000]
val_images = scale_images(val_images,(299,299,3))
final_images = np.random.randn(1000,32)
final_images = SAGAN_GENERATOR.predict(final_images)
final_images = [(image + 1)/2 for image in final_images]
final_images = scale_images(final_images,(299,299,3))
fid_score = calculate_fid(model,val_images,final_images)
print("FID score of SAGAN is ",fid_score)
