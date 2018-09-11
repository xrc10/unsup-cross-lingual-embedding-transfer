import tensorflow as tf

from model.common_layers import *

def gradient_norm_from_one(d_hat, x_hat, scale):
    ddx = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
    return ddx

def set_W_gan_layers(true_tensor, fake_tensor, input_dim, disc_hidden_dim, scale, name):
    disc_input = tf.concat([true_tensor, fake_tensor], 0)
    disc_logits, Ws, bs = dense_layers(input_layer=disc_input, input_dim=input_dim, hidden_dim=disc_hidden_dim, output_dim=1, activation='tanh', add_multiple_noise=0.5, name=name, BN=False)
    gen_logits, _, _ = dense_layers(input_layer=fake_tensor, input_dim=input_dim, hidden_dim=disc_hidden_dim, output_dim=1, Ws=Ws, bs=bs, activation='tanh', name=name, BN=False)
    gen_label = tf.ones([tf.shape(fake_tensor)[0]], dtype=tf.float32)
    disc_label = tf.concat([tf.ones([tf.shape(true_tensor)[0]], dtype=tf.float32),-tf.ones([tf.shape(fake_tensor)[0]], dtype=tf.float32)], 0)
    l_disc = tf.reduce_mean(tf.multiply(tf.squeeze(disc_logits), disc_label))
    l_gen = tf.reduce_mean(tf.multiply(tf.squeeze(gen_logits), gen_label))

    # Improved Training of Wasserstein GANs
    epsilon = tf.random_uniform(tf.shape(true_tensor), 0.0, 1.0)
    tensor_hat = epsilon * fake_tensor + (1 - epsilon) * true_tensor
    # print('tensor_hat.get_shape()', tensor_hat.get_shape())
    logits_hat, _, _ = dense_layers(input_layer=tensor_hat, input_dim=input_dim, hidden_dim=disc_hidden_dim, output_dim=1, Ws=Ws, bs=bs, activation='tanh', name=name, BN=False)
    ddx = gradient_norm_from_one(logits_hat, tensor_hat, scale)
    # print('ddx.get_shape()', ddx.get_shape())
    return l_disc+ddx, l_gen, Ws, bs
