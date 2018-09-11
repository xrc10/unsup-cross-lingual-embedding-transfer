import tensorflow as tf
from model.distance import l1_norm

def get_sinkhorn_distance(M, r, c, lambda_sh, depth):
    n = tf.shape(M)[0]

    # r = tf.ones([n, 1], tf.float32)/tf.cast(n, dtype=tf.float32)
    # c = tf.ones([n, 1], tf.float32)/tf.cast(n, dtype=tf.float32)

    r = tf.expand_dims(l1_norm(r), 1)
    c = tf.expand_dims(l1_norm(c), 1)

    K = tf.exp(-lambda_sh*M)
    K_T = tf.transpose(K)
    v = tf.ones([n, 1], tf.float32)/tf.cast(n, dtype=tf.float32)

    for i in range(depth):
        u = r/(tf.matmul(K,v))
        v = c/(tf.matmul(K_T,u))

    return tf.reduce_sum(tf.multiply(u,tf.matmul(tf.multiply(K,M),v)))