
import tensorflow as tf

class Layer(object):
    
    def __init__(self, layer_name, paddingMethod="SAME"):
        
        self.layer_name = layer_name
        self.paddingMethod=paddingMethod
    
    # padding="SAME" or "VALID"
    def _conv_relu(self, input_tensor, weight_shape, bias_shape):
        
        self.weight = tf.get_variable("weight", weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        self.bias = tf.get_variable("bias", bias_shape, initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 1, 1, 1], padding=self.paddingMethod)
        
        return tf.nn.relu(conv + self.bias)
    
    def convolve(self, input_tensor, weight_shape, bias_shape):
        with tf.variable_scope(self.layer_name):
            relu = self._conv_relu(input_tensor, weight_shape, bias_shape)
            return relu
        
    # input batchx13x13x80
    def one_filter_out(self, input_tensor, boardsize):
        input_shape = input_tensor.get_shape()
        batch_size = input_shape[0].value
        assert(input_shape[1] == boardsize)
        assert(input_shape[2] == boardsize)
        print("input_shape ", input_shape)
        weight_shape = (1, 1, input_shape[3], 1)
        bias_shape = (boardsize * boardsize)
        
        self.weight = tf.get_variable("output_layer_weight", weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        self.bias = tf.get_variable("position_bias", bias_shape, initializer=tf.constant_initializer(0.0))
        
        out = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 1, 1, 1], padding="SAME")
        logits = tf.reshape(out, shape=(batch_size, boardsize * boardsize)) + self.bias
        
        return logits
        
    # input 13x13xchanels, VALID padding, 1x1 convolution, output is 13x13, then softmax
    # position bias shape [13*13]
    # filter shape 1x1x1, only 1 filter
    def policy_network_out(self, input_tensor, weight_shape, bias_shape):
        self.weight = tf.get_variable("output_layer_weight", weight_shape, initializer=tf.random_uniform_initializer())
        
        self.bias = tf.get_variable("position_bias", bias_shape, initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 1, 1, 1], padding="VALID")
        
        prob = tf.nn.softmax(tf.squeeze(conv) + self.bias)
        
        return prob
    
        
if __name__ == "__main__":
    sess = tf.InteractiveSession()
    conv1 = Layer("layer1")
    # x_in=tf.placeholder(dtype=tf.float32, shape=[1,227,227,3])
    x_in = tf.Variable(tf.random_normal([1, 227, 227, 3]))
    out = conv1.convolve(x_in, weight_shape=[11, 11, 3, 96], bias_shape=[96])

    print(out)
    print(out.get_shape())
    sp = out.get_shape()
    print(sp[2])

