from tensorflow.python.framework import function
import tensorflow as tf

ETA=1e-8

## We all assume that:
# a: n x d matrix
# b: n x d matrix
# distance before being normalized by n or d

# @function.Defun(tf.float32, tf.float32)
# def norm_grad(x, dy):
#     return dy*(x/tf.norm(x))

# @function.Defun(tf.float32, grad_func=norm_grad)
def norm(x):
    return tf.norm(x, axis=1, keep_dims=True)

def l2_norm(a):
    norm_a = tf.sqrt(tf.reduce_sum(tf.square(a), 1, keep_dims=True)+ETA)
    # norm_a = norm(a)
    normalize_a = a/(norm_a)
    return normalize_a

def l1_norm(a):
    norm_a = tf.reduce_sum(a, keep_dims=True)
    normalize_a = a/(norm_a)
    return normalize_a

def cosine_similarity(a, b):
    """ 
    compute cosine similarity between v1 and v2
    """
    normalize_a = l2_norm(a)      
    normalize_b = l2_norm(b)
    return tf.reduce_sum(tf.multiply(normalize_a,normalize_b))

def cosine_similarity_matrix(a, b):
    """ 
    compute cosine similarity between v1 and v2 in matrix
    """
    normalize_a = l2_norm(a)      
    normalize_b = l2_norm(b)
    return tf.matmul(normalize_a, tf.transpose(normalize_b))

def euclidean_distance(a, b):
    # return tf.reduce_sum(tf.norm(tf.subtract(a, b), ord='euclidean', axis=1))
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)+ETA))

def euclidean_distance_matrix(a, b):
    batch_size = tf.shape(a)[0]
    aaT = tf.matmul(a, tf.transpose(a))
    bbT = tf.matmul(b, tf.transpose(b))
    l2 = tf.maximum(aaT + bbT - 2*tf.matmul(a, tf.transpose(b)), ETA*tf.ones([batch_size,batch_size], tf.float32))
    return tf.sqrt(l2)

# def euclidean_distance_matrix(a, b):
#     # assuming a and b are normalized
#     return tf.sqrt(2 - 2*tf.matmul(a, tf.transpose(b)))

def norm_euclidean_distance(a, b):
    normalize_a = l2_norm(a)      
    normalize_b = l2_norm(b)
    return euclidean_distance(normalize_a, normalize_b)

def norm_euclidean_distance_matrix(a, b):
    return tf.sqrt(2 - 2*cosine_similarity_matrix(a,b))

