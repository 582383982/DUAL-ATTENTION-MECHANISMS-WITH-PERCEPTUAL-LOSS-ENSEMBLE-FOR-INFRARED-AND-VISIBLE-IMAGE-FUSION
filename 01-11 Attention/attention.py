import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import cv2
WEIGHT_INIT_STDDEV = 0.1
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

class Attention(object):
    def __init__(self, model_pre_path_a):

        self.weights = []
        self.model_pre_path = model_pre_path_a

        with tf.variable_scope('get_attention'):
            self.weights.append(self._create_variables(32, 4, 1, scope='attention_block_conv1'))
            self.weights.append(self._create_variables(32, 32, 1, scope='attention_block_conv2'))
            # self.weights.append(self._create_variables(4, 8, 3, scope='attention_block_conv3'))
            # self.weights.append(self._create_variables(8, 16, 3, scope='attention_block_conv4'))
            # self.weights.append(self._create_variables(16, 32, 3, scope='attention_block_conv5'))
            # self.weights.append(self._create_variables(32, 64, 3, scope='attention_block_conv6'))
            # self.weights.append(self._create_variables(64, 64, 1, scope='attention_block_conv7'))

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
                #第一个参数是张量的维度，第二个是标准差
                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)
    def get_attention(self, image):
        inputs_shape = image.get_shape().as_list()
        batchsize, height, width, C = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]
        # for i in range(len(self.weights)):
        #     kernel, bias = self.weights[i]
        #     if i == 6:
        #         out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding='VALID')
        #         out = tf.nn.bias_add(out, bias)
        #         out = tf.nn.relu(out)
        #     elif i % 2 == 0:
        #         out = conv2d(out, kernel, bias, use_relu=True)
        #     else:
        #         out = tf.nn.atrous_conv2d(out, filters=kernel, rate=2, padding='SAME')
        kernel, bias = self.weights[0]
        query_conv = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
        query_conv = tf.nn.bias_add(query_conv, bias)
        query_conv = tf.nn.relu(query_conv)

        key_conv = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
        key_conv = tf.nn.bias_add(key_conv, bias)
        key_conv = tf.nn.relu(key_conv)

        kernel, bias = self.weights[1]
        value_conv = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
        value_conv = tf.nn.bias_add(value_conv, bias)
        value_conv = tf.nn.relu(value_conv)

        proj_query = tf.reshape(query_conv, [batchsize, width * height, -1])
        proj_key = tf.transpose((tf.reshape(key_conv, [batchsize, width * height, -1])), perm=[0, 2, 1])
        energy = tf.matmul(proj_query, proj_key)

        attention = tf.nn.softmax(energy)
        proj_value = tf.reshape(value_conv, [batchsize, width * height, -1])

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, [batchsize, height, width, C])
        out = out + image

        return out

    def guideFilter(self, I, p, winSize, eps):

        mean_I = cv2.blur(I, winSize)  # I的均值平滑
        mean_p = cv2.blur(p, winSize)  # p的均值平滑

        mean_II = cv2.blur(I * I, winSize)  # I*I的均值平滑
        mean_Ip = cv2.blur(I * p, winSize)  # I*p的均值平滑

        var_I = mean_II - mean_I * mean_I  # 方差
        cov_Ip = mean_Ip - mean_I * mean_p  # 协方差

        a = cov_Ip / (var_I + eps)  # 相关因子a
        b = mean_p - a * mean_I  # 相关因子b

        mean_a = cv2.blur(a, winSize)  # 对a进行均值平滑
        mean_b = cv2.blur(b, winSize)  # 对b进行均值平滑

        q = mean_a * I + mean_b

        return q

    def RollingGuidance(self,I, sigma_s, sigma_r, iteration):
        sigma_s = (sigma_s, sigma_s)
        out = cv2.GaussianBlur(I, sigma_s, 0)
        sigma_r = sigma_r*sigma_r
        for i in range(iteration):
            out = self.guideFilter(out, I, sigma_s, sigma_r)

        return out

    def Grad(self,I1):
        G1 = []
        L1 = []
        G1.append(I1)
        sigma_s = 3
        sigma_r = [0.5, 0.5, 0.5, 0.5]
        iteration = [3, 3, 3, 3]
        indice = (1, 2, 3)
        for i in indice:
            G1.append(self.RollingGuidance(G1[i - 1], sigma_s, sigma_r[i - 1], iteration[i - 1]))
            L1.append(G1[i - 1] - G1[i])
            sigma_s = 3 * sigma_s
        sigma_s = (3, 3)
        G1.append(cv2.GaussianBlur(G1[3], sigma_s, 0))
        L1.append(G1[3] - G1[4])
        L1.append(G1[4])
        grad = L1[0]
        return grad
