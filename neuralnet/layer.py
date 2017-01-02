
import tensorflow as tf

class Layer(object):
    # paddingMethod="SAMVE" then the output is the same size, or VALID uses ordinary padding
    # reuse_var=True then try to reuse created variables with the same scoped name
    def __init__(self, layer_name, paddingMethod="SAME", reuse_var=False):
        self.layer_name = layer_name
        self.paddingMethod=paddingMethod
        self.reuse_var=reuse_var

    # padding="SAME" or "VALID"
    def _conv2d(self, input_tensor, weight_shape, bias_shape):
        
        self.weight = tf.get_variable("weight", weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        self.bias = tf.get_variable("bias", bias_shape, initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 1, 1, 1], padding=self.paddingMethod)
        
        return (conv + self.bias)

    def _relu(self, input_tensor):
        return tf.nn.relu(input_tensor)

    def convolve(self, input_tensor, weight_shape, bias_shape):
        with tf.variable_scope(self.layer_name) as sp:
            if self.reuse_var:
                sp.reuse_variables()
            relu=self._relu(self._conv2d(input_tensor, weight_shape, bias_shape))
            return relu

    def convolve_no_relu(self, input_tensor, weight_shape, bias_shape):
        with tf.variable_scope(self.layer_name) as sp:
            if self.reuse_var:
                sp.reuse_variables()
            return self._conv2d(input_tensor, weight_shape, bias_shape)

    #logits for move prediction
    def move_logits(self, input_tensor, boardsize, value_net=False):
        with tf.variable_scope(self.layer_name) as sp:
            if self.reuse_var:
                sp.reuse_variables()
            return self._one_filter_out(input_tensor, boardsize, value_net)

    # input batchsize x BOARDSIZE x BOARDSIZE x DEPTH
    def _one_filter_out(self, input_tensor, boardsize, value_net=False):
        input_shape = input_tensor.get_shape()
        batch_size = input_shape[0].value
        assert(input_shape[1] == boardsize)
        assert(input_shape[2] == boardsize)
        weight_shape = (1, 1, input_shape[3], 1)
        bias_shape = (boardsize * boardsize)
        
        self.weight = tf.get_variable("output_layer_weight", weight_shape,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.bias = tf.get_variable("position_bias", bias_shape, initializer=tf.constant_initializer(0.0))
        
        out = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 1, 1, 1], padding="SAME")

        if value_net: return out

        logits = tf.add(tf.reshape(out, shape=(batch_size, boardsize * boardsize)), self.bias, name="output_node")
        return logits

    #output for the value net, one tanh unit fully-connected to the previous layer
    #two layers, one full-size convolution follows with num_units relu, another convolution follows 1 tanh
    def value_estimation(self, input_tensor, num_units):
        with tf.variable_scope(self.layer_name) as sp:
            if self.reuse_var:
                sp.reuse_variables()
            assert(self.paddingMethod=="VALID")
            out1=self._fully_connected(input_tensor, num_units, "relu")
        with tf.variable_scope(self.layer_name+"_2"):
            #out2=self._dropout(out1, keep_prob)
            out3=self._fully_connected(out1, 1, "sigmoid")
            return tf.squeeze(out3, name="value_output_node")

    #fully-connected layer can be seen as convolutions: num_units is number of filters,
    #kernal size is equal to (input_width, input_height)
    def _fully_connected(self, input_tensor, num_units, unit_type):
        input_shape=input_tensor.get_shape()
        weight_shape=(input_shape[1], input_shape[2], input_shape[3], num_units)
        bias_shape=(num_units,)
        out=self._conv2d(input_tensor, weight_shape, bias_shape)
        if unit_type == "sigmoid":
            return tf.sigmoid(out)
            #return tf.tanh(out)
        elif unit_type =="relu":
            return tf.nn.relu(out)

    def _dropout(self, input_tensor, keep_prob):
        return tf.nn.dropout(input_tensor, keep_prob)

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

