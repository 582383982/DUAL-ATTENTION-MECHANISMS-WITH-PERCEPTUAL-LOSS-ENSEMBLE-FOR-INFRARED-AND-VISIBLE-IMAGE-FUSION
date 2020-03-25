import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

WEIGHT_INIT_STDDEV = 0.1


class Decoder(object):

    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.variable_scope('decoder'):

            self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_1'))
            self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(48, 16, 3, scope='conv2_3'))
            self.weight_vars.append(self._create_variables(16, 1 , 3, scope='conv2_4'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):

        if self.model_pre_path:
            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('decoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('decoder/' + scope + '/bias'), name='bias')
        else:
            with tf.variable_scope(scope):
                shape = [kernel_size, kernel_size, input_filters, output_filters]
                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def decode(self, image,block,block2):
        final_layer_idx  = len(self.weight_vars) - 1


        out = image
        print("1111111111111111111111111111")
        print(out.shape)
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                if i==2:
                    out = conv2d(out, kernel, bias)
                    out = up_sample(out, scale_factor = 2)
                elif i==1:
                    out = conv2d(out, kernel, bias)
                    out = up_sample(out, scale_factor=2)
                    print("===================================")
                    print(out.shape)
                    print(block.shape)
                    out = tf.concat([out, block], 3)

                else:
                    out = conv2d(out, kernel, bias)
                    out = up_sample(out, scale_factor=2)
                    print("===================================")
                    print(out.shape)
                    print(block2.shape)
                    out = tf.concat([out, block2], 3)
            # print('decoder ', i)
            # print('decoder out:', out.shape)
        return out

def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)

def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out

