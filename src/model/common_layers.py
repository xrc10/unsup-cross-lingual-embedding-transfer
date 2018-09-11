import tensorflow as tf

USE_BN = True

def highway_layer(input_layer, input_dim, hidden_dim, output_dim, Wg=None, bg=None, Ws=None, bs=None, name='highway'):
    with tf.name_scope(name + "-gate-layer"):
        if Wg is None:
            Wg = tf.Variable(tf.truncated_normal([input_dim, input_dim], stddev=0.05), name="W")
        if bg is None:
            bg = tf.Variable(tf.truncated_normal([input_dim], stddev=0.05), name="b")
        g = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, Wg), bg))

    trans, Ws, bs = dense_layers(input_layer, input_dim, hidden_dim, output_dim, Ws=Ws, bs=bs, name=name+'-dense-layer')
    out = tf.multiply(g, trans) + tf.multiply(1-g, input_layer)
    return out, Wg, bg, Ws, bs

def dense_layers(input_layer, input_dim, hidden_dim, output_dim, Ws=None, bs=None, name='fc', activation='elu', bias=True, orth_init=False, add_multiple_noise=None, BN=USE_BN, BN_phase=True, BN_reuse=None):
    if Ws is None:
        Ws = (len(hidden_dim)+1)*[None]
    if bs is None:
        bs = (len(hidden_dim)+1)*[None]
    dims = [input_dim] + hidden_dim + [output_dim]
    layers = [input_layer] + len(hidden_dim)*[None] + [None]
    with tf.name_scope(name):
        for i, d_i in enumerate(dims[:-1]):
            # add multiple noise to input layers
            if add_multiple_noise is not None:
                # print("add_multiple_noise", add_multiple_noise)
                gaussian_noise = tf.random_normal(shape = tf.shape(layers[i]), mean = 1.0, stddev = add_multiple_noise, dtype = tf.float32)
                layers[i] = tf.multiply(layers[i], gaussian_noise)
            # Wx + b
            d_o = dims[i+1]
            if Ws[i] is None:
                if orth_init:
                    initializer = tf.orthogonal_initializer
                else:
                    initializer = tf.contrib.layers.xavier_initializer()
                Ws[i] = tf.get_variable(name="W"+str(i)+name, shape=[d_i, d_o], dtype=tf.float32, initializer=initializer)
            if bs[i] is None and bias:
                bs[i] = tf.Variable(tf.constant(0.1, shape=[d_o], dtype=tf.float32), name="b"+str(i))
            elif not bias:
                bs[i] = tf.constant(0, shape=[d_o], dtype=tf.float32)

            layers[i+1] = tf.nn.xw_plus_b(layers[i], Ws[i], bs[i], name="fc_out"+str(i+1))

            if BN and i!=len(hidden_dim): # add BN
            # if BN:
                layers[i+1] = tf.contrib.layers.batch_norm(layers[i+1], center=True, scale=True, is_training=BN_phase, scope=name+'_bn_'+str(i), reuse=BN_reuse)

            if i!=len(hidden_dim) and activation:
                # layers[i+1] = tf.nn.tanh(layers[i+1]) # tanh activation for any other output
                if activation == 'elu':
                    layers[i+1] = tf.nn.elu(layers[i+1]) # elu activation for any other output 
                elif activation == 'relu':
                    layers[i+1] = tf.nn.relu(layers[i+1]) # relu activation for any other output
                elif activation == 'tanh':
                    layers[i+1] = tf.nn.tanh(layers[i+1]) # tanh activation for any other output

    return layers[len(hidden_dim)+1], Ws, bs 

def add_multiple_gaussian_noise(inp_tensor, stddev):
    gaussian_noise = tf.random_normal(shape = tf.shape(inp_tensor), mean = 1.0, stddev = stddev, dtype = tf.float32)
    return tf.multiply(inp_tensor, gaussian_noise)