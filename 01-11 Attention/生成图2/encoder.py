import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import cv2
import  attention
WEIGHT_INIT_STDDEV = 0.1
DENSE_layers = 3
DECAY = .9
EPSILON = 1e-8
class Encoder(object):
    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.variable_scope('encoder'):
            self.weight_vars.append(self._create_variables(1, 16, 3, scope='conv1_1'))
#---------------------------------------------------------------------------------------------------
            self.weight_vars.append(self._create_variables(16, 16, 3, scope='dil_block_conv1'))
            self.weight_vars.append(self._create_variables(16, 32, 3, scope='dil_block_conv2'))
            self.weight_vars.append(self._create_variables(32, 64, 3, scope='dil_block_conv3'))

            self.weight_vars.append(self._create_variables(16, 16, 3, scope='dil_block_conv4'))
            self.weight_vars.append(self._create_variables(16, 32, 3, scope='dil_block_conv5'))
            self.weight_vars.append(self._create_variables(32, 64, 3, scope='dil_block_conv6'))

            self.weight_vars.append(self._create_variables(16, 16, 3, scope='dil_block_conv7'))
            self.weight_vars.append(self._create_variables(16, 32, 3, scope='dil_block_conv8'))
            self.weight_vars.append(self._create_variables(32, 64, 3, scope='dil_block_conv9'))
            # self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv1_2'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        # 3 * 3 * input * output
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        if self.model_pre_path:

            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('encoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('encoder/' + scope + '/bias'), name='bias')
        else:
            with tf.variable_scope(scope):
                # truncated_normal 从截断的正态分布中输出随机值
                # 第一个参数是张量的维度，第二个是标准差

                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def encode(self, image):
        dia_indices_1 = (1, 2, 3)
        dia_indices_2 = (4, 5, 6)
        dia_indices_3 = (7, 8, 9)
        res_block=[]

        #out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            #filter= tf.constant(value=1, shape=[3, 3, 16, 16], dtype=tf.float32)
            if i==0:
                former = conv2d(image, kernel, bias, use_relu=True)
                res_block.append(former)                                                            # 0
            if i in dia_indices_1:
                print("----------------------------------")
                print(i)
                if(i==1):
                    x = tf.nn.atrous_conv2d(former, filters=kernel,rate=1,padding='SAME')
                    x = tf.nn.bias_add(x, bias)
                    x = tf.nn.relu(x)
                    x = tf.nn.max_pool(x, 2, [1,2,2,1], 'VALID', name=None)
                    res_block.append(x)                                                             # 1    16

                elif(i==2):
                    y=tf.nn.atrous_conv2d(x, filters=kernel,rate=1,padding='SAME')
                    y = tf.nn.bias_add(y, bias)
                    y = tf.nn.relu(y)
                    y = tf.nn.max_pool(y, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(y)   # 2  32
                    print("eeeeeeeeeeeeeeeeeeeeeeeeeee")
                    print(y.shape)
                    print(res_block[2].shape)


                else:
                    z = tf.nn.atrous_conv2d(y, filters=kernel, rate=1, padding='SAME')
                    z = tf.nn.bias_add(z, bias)
                    z = tf.nn.relu(z)
                    z = tf.nn.max_pool(z, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(z)                                                            # 3   64


            if i in dia_indices_2:
                print("----------------------------------")
                print(i)
                if (i == 4):
                    x = tf.nn.atrous_conv2d(former, filters=kernel, rate=2, padding='SAME')
                    x = tf.nn.bias_add(x, bias)
                    x = tf.nn.relu(x)
                    x = tf.nn.max_pool(x, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(x)                                  #4
                elif (i == 5):
                    y = tf.nn.atrous_conv2d(x, filters=kernel, rate=2, padding='SAME')
                    y = tf.nn.bias_add(y, bias)
                    y = tf.nn.relu(y)
                    y = tf.nn.max_pool(y, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(y)                            #5
                else:
                    z = tf.nn.atrous_conv2d(y, filters=kernel, rate=2, padding='SAME')
                    z = tf.nn.bias_add(z, bias)
                    z = tf.nn.relu(z)
                    z = tf.nn.max_pool(z, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(z)                                                           # 6


            if i in dia_indices_3:
                print("----------------------------------")
                print(i)
                if (i == 7):
                    x = tf.nn.atrous_conv2d(former, filters=kernel, rate=4, padding='SAME')
                    x = tf.nn.bias_add(x, bias)
                    x = tf.nn.relu(x)
                    x = tf.nn.max_pool(x, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(x)                                #7
                elif (i == 8):
                    y = tf.nn.atrous_conv2d(x, filters=kernel, rate=4, padding='SAME')
                    y = tf.nn.bias_add(y, bias)
                    y = tf.nn.relu(y)
                    y = tf.nn.max_pool(y, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(y)                   #8
                else:
                    z = tf.nn.atrous_conv2d(y, filters=kernel, rate=4, padding='SAME')
                    z = tf.nn.bias_add(z, bias)
                    z = tf.nn.relu(z)
                    z = tf.nn.max_pool(z, 2, [1, 2, 2, 1], 'VALID', name=None)
                    res_block.append(z)                                                          # 9





        out = 1 * res_block[3] + 1 * res_block[6] + 1 * res_block[9]

        block = res_block[1]+res_block[4]+res_block[7]       #16
        block2 = res_block[2]+res_block[5]+res_block[8]        #32

        block = CAM_module(block)
        block2 = CAM_module(block2)

        print(self.weight_vars[0])

        return out,res_block,block,block2



#---------------------------------------------------------------------------------
# x : 输入
# kernel, bias : 卷积核, 偏移量
# use_relu : 激活
def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    # num_maps = x_padded.shape[3]
    # out = __batch_normalize(x_padded, num_maps)
    # out = tf.nn.relu(out)
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)
    return out


def transition_block(x, kernel, bias):

    num_maps = x.shape[3]
    out = __batch_normalize(x, num_maps)
    out = tf.nn.relu(out)
    out = conv2d(out, kernel, bias, use_relu=False)

    return out


def __batch_normalize(inputs, num_maps, is_training=True):
    # Trainable variables for scaling and offsetting our inputs
    # scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32))
    # offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32))

    # Mean and variances related to our current batch
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

    # # Create an optimizer to maintain a 'moving average'
    # ema = tf.train.ExponentialMovingAverage(decay=DECAY)
    #
    # def ema_retrieve():
    #     return ema.average(batch_mean), ema.average(batch_var)
    #
    # # If the net is being trained, update the average every training step
    # def ema_update():
    #     ema_apply = ema.apply([batch_mean, batch_var])
    #
    #     # Make sure to compute the new means and variances prior to returning their values
    #     with tf.control_dependencies([ema_apply]):
    #         return tf.identity(batch_mean), tf.identity(batch_var)
    #
    # # Retrieve the means and variances and apply the BN transformation
    # mean, var = tf.cond(tf.equal(is_training, True), ema_update, ema_retrieve)
    bn_inputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, None, None, EPSILON)

    return bn_inputs

def CAM_module(inputs):
    inputs_shape = inputs.get_shape().as_list()
    batchsize, height, width, C = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]

    proj_query = tf.transpose(tf.reshape(inputs, [batchsize, width*height, -1]), perm=[0, 2, 1])
    proj_key = tf.reshape(inputs, [batchsize, width*height, -1])
    energy = tf.matmul(proj_query, proj_key)
    energy_new = tf.maximum(energy, -1)-energy

    attention = tf.nn.softmax(energy_new)
    proj_value = tf.transpose(tf.reshape(inputs, [batchsize, width * height, -1 ]), perm=[0, 2, 1])

    out = tf.transpose(tf.matmul(attention, proj_value), perm=[0, 2, 1])
    out = (tf.reshape(out, [batchsize, height, width, C]))
    out = out + inputs
    return out