#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import errno

import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.python import debug as tf_debug

from utils.callbacks import SGDWR, EarlyStopper, ModelChecker
from utils.eval_helper import knn_accuracy_from_list, find_common_words
from utils.visual_emb import plot_align_matrix, plot_emb
from utils.data_helper import load_text_vec, load_freq, uniform_batch_iter, freq_sample_batch_iter, sparsify_mat, mkdir_p

from model.distance import *
from model.common_layers import *
from model.cycle_emb import CycleEmb, USE_EUC_DIST, TIED_WEIGHTS
from model.sinkhorn import get_sinkhorn_distance
from model.wasserstein_gan import set_W_gan_layers

from numpy import linalg as LA
from functools import partial
import time
import warnings

USE_EUC_DIST_MATRIX = False
TRANS_ITER = 1
DISC_ITERS = 5 # for WGAN, we iterate discriminator
WGAN_TRANS_ITERS = 5 # for WGAN, we iterate transformation
DEBUG = False
LAMBDA_SH = 10
SINKHORN_LAYER_DEPTH = 20
MULTIPLE_NOISE_STD = None # add noise to sinkhorn layers
PATIENCE = 4000 # number of epochs we wait before call early stop
MIN_DELTA = 1e-5 # improvement larger than delta is considered as improvement

################################################################################
### The cross-lingual embeding model learned from cycle adversarial network
class CycleAlignEmb(CycleEmb):
    def __init__(self, trans_hidden_dim=[50], disc_hidden_dim=[50], recon_loss_weight=[1,1], constraint_loss_weight=1, init_align_loss_weight=1, trans_activation=False, lambda_sh=10, sinkhorn_layer_depth=20, cyc_loss=True, use_sinkhorn=True, use_BN=True, norm='scale', save_path=None, F_init=None):
        CycleEmb.__init__(self, trans_hidden_dim, disc_hidden_dim, recon_loss_weight, trans_activation, cyc_loss, use_sinkhorn, use_BN, norm, save_path)
        self.constraint_loss_weight = constraint_loss_weight # relative weight of loss of F constraint
        self.init_align_loss_weight = init_align_loss_weight # if we have initial seed, the weight of that part of loss
        self.lambda_sh = lambda_sh
        self.sinkhorn_layer_depth = sinkhorn_layer_depth
        self.F_init = F_init
        self.cyc_loss = cyc_loss

    def build_WGAN_layer(self, ):
        if self.F_init is None:
            l_disc_s, l_gen_s, self.Ws_s_vs_fs, self.bs_s_vs_fs = set_W_gan_layers(self.input_s, self.trans_t, input_dim=self.emb_dim_list[0], disc_hidden_dim=self.disc_hidden_dim, scale=10, name='W_gan_s')
            l_disc_t, l_gen_t, self.Ws_t_vs_ft, self.bs_t_vs_ft = set_W_gan_layers(self.input_t, self.trans_s, input_dim=self.emb_dim_list[1], disc_hidden_dim=self.disc_hidden_dim, scale=10, name='W_gan_t')
            if self.cyc_loss:
                self.l_disc = (l_disc_s + l_disc_t)/2
                self.l_gen = (l_gen_s + l_gen_t)/2
            else:
                self.l_disc = l_disc_s
                self.l_gen = l_gen_s
        else:
            self.l_disc = tf.constant(0, dtype=tf.float32)
            self.l_gen = tf.constant(0, dtype=tf.float32)

    def build_sinkhorn_layer(self, lambda_sh=LAMBDA_SH, sinkhorn_layer_depth=SINKHORN_LAYER_DEPTH, multiple_noise_std=None):
        '''
        Following "Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport, NIPS 2013"
        '''
        if multiple_noise_std is not None:
            self.input_t_inp = add_multiple_gaussian_noise(self.input_t, multiple_noise_std)
            self.trans_s_inp = add_multiple_gaussian_noise(self.trans_s, multiple_noise_std)
            self.trans_t_inp = add_multiple_gaussian_noise(self.trans_t, multiple_noise_std)
            self.input_s_inp = add_multiple_gaussian_noise(self.input_s, multiple_noise_std)
        else:
            self.input_t_inp = self.input_t
            self.trans_s_inp = self.trans_s
            self.trans_t_inp = self.trans_t
            self.input_s_inp = self.input_s

        M1 = euclidean_distance_matrix(self.trans_s_inp, self.input_t_inp) if USE_EUC_DIST_MATRIX else norm_euclidean_distance_matrix(self.trans_s_inp, self.input_t_inp)
        M2 = euclidean_distance_matrix(self.input_s_inp, self.trans_t_inp) if USE_EUC_DIST_MATRIX else norm_euclidean_distance_matrix(self.input_s_inp, self.trans_t_inp)
        
        if self.cyc_loss:
            self.sinkhorn_distance = (get_sinkhorn_distance(M1, self.freq_s, self.freq_t, lambda_sh=lambda_sh, depth=sinkhorn_layer_depth) + \
                                  get_sinkhorn_distance(M2, self.freq_s, self.freq_t, lambda_sh=lambda_sh, depth=sinkhorn_layer_depth))/2
        else:
            self.sinkhorn_distance = get_sinkhorn_distance(M2, self.freq_s, self.freq_t, lambda_sh=lambda_sh, depth=sinkhorn_layer_depth)

    def build_F_init_layer(self):
        if self.F_init is not None:
            t_nz_idx, s_nz_idx = np.where(self.F_init>0)
            t_nz_idx, s_nz_idx = np.sort(t_nz_idx), np.sort(s_nz_idx)
            self.F_init = self.F_init[t_nz_idx,:]
            self.F_init = self.F_init[:,s_nz_idx]
            self.F_init_row=self.F_init/np.sum(self.F_init, axis=1, keepdims=True)
            self.F_init_col=np.transpose(self.F_init/np.sum(self.F_init, axis=0, keepdims=True))
            # print('self.F_init', self.F_init)
            # print('self.F_init.shape', self.F_init.shape)
            # take care of the initial dictionary
            # F_init_count = np.count_nonzero(self.F_init)
            # print('F_init_count', F_init_count)
            F_init_row = tf.convert_to_tensor(self.F_init_row, dtype=tf.float32)
            F_init_col = tf.convert_to_tensor(self.F_init_col, dtype=tf.float32)

            self.init_vecs_list = [None, None]
            self.init_vecs_list[0] = self.vecs_list[0][s_nz_idx,:]
            self.init_vecs_list[1] = self.vecs_list[1][t_nz_idx,:]

            if USE_EUC_DIST:
                self.align_loss_row_init = euclidean_distance(self.trans_s_init, self.input_t_init) / tf.cast(self.F_init.shape[0], dtype=tf.float32)
                self.align_loss_col_init = euclidean_distance(self.trans_t_init, self.input_s_init) / tf.cast(self.F_init.shape[1], dtype=tf.float32)
            else:
                self.align_loss_row_init = 1 - cosine_similarity(tf.matmul(F_init_row, self.trans_s_init), self.input_t_init) / tf.cast(self.F_init.shape[0], dtype=tf.float32)
                self.align_loss_col_init = 1 - cosine_similarity(tf.matmul(F_init_col, self.trans_t_init), self.input_s_init) / tf.cast(self.F_init.shape[1], dtype=tf.float32)
            self.init_align_loss = (self.align_loss_row_init + self.align_loss_col_init)/2
        else:
            self.init_align_loss = tf.constant(0, dtype=tf.float32)

    def build_model(self):
        """ 
        build the graph for training
        internal dimentions:
        bs: batch_size, ed: embedding dimension, hd: hidden dimension
        """
        self.build_input_layers()
        self.build_transfer_layers()

        # alignment loss
        self.build_sinkhorn_layer(lambda_sh=self.lambda_sh, sinkhorn_layer_depth=self.sinkhorn_layer_depth, multiple_noise_std=MULTIPLE_NOISE_STD)
        self.build_F_init_layer()
        self.build_WGAN_layer()

        # recontruction loss
        self.build_recontruction_layers()
        self.l_trans_wgan = self.l_gen + self.recon_loss_weight[0]*self.reconstruct_loss
        self.l_trans_sh = self.sinkhorn_distance + self.recon_loss_weight[1]*self.reconstruct_loss + self.init_align_loss_weight*self.init_align_loss

        self.set_summary()

    def set_summary(self):
        tf.summary.scalar('discriminator loss', self.l_disc)
        tf.summary.scalar('wgan transformation(generator) loss', self.l_trans_wgan)
        tf.summary.scalar('sinkhorn transformation loss', self.l_trans_sh)
        tf.summary.scalar('reconstruct loss', self.reconstruct_loss)
        tf.summary.scalar('init alignment loss', self.init_align_loss)
        tf.summary.scalar('sinkhorn distance', self.sinkhorn_distance)
        if len(self.trans_hidden_dim) == 0:
            W = self.Ws_s2t[0]
            self.orth_loss = tf.norm(tf.matmul(W, W, transpose_a=True)-tf.eye(self.emb_dim_list[0]), ord='fro', axis=(0,1))
            tf.summary.scalar('orth_loss', self.orth_loss)
        self.merged_summary = tf.summary.merge_all()

    def train(self, batch_size=128, lr=0.001, epoch=10000, wgan_epoch=2000, logger=None, validation=None, dp_keep_prob=1.0, patience=PATIENCE, min_delta = MIN_DELTA):
        """ 
        train with adversarial/sinkhorn loss
        log: log file to record training
        validation: [tgt_words, true_src_words]
        """
        batch_per_epoch = int(self.vecs_list[0].shape[0]/batch_size) + 1 # how many steps in a epoch
        total_steps = batch_per_epoch * epoch
        early_stopper = EarlyStopper(patience*batch_per_epoch, min_delta)
        model_checker = ModelChecker()

        learning_rate = tf.placeholder(tf.float32, shape=[])
        lr_schedule_dict = SGDWR(T_total=epoch, T_0=10, T_mult=1.2, lr_max=lr, lr_min=lr/100)

        if self.F_init is None: # we will switch from WGAN to Sinkhorn during training
            swith_flag = False # first WGAN and then Sinkhorn
            # swith_flag = True # Use Sinkhorn all the time

        with tf.Session(config=self.session_conf) as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            self.writer = tf.summary.FileWriter(os.path.join(self.save_path, 'graph'), graph=sess.graph)
            if self.saver is None:
                self.saver = tf.train.Saver(tf.global_variables())
                # self.saver = tf.train.Saver(trans_variables)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            
            # define train operation
            if TIED_WEIGHTS:
                trans_variables = self.Ws_s2t
            else:
                trans_variables = self.Ws_s2t + self.Ws_t2s
            if self.F_init is None:
                disc_variables = self.Ws_s_vs_fs + self.bs_s_vs_fs + self.Ws_t_vs_ft + self.bs_t_vs_ft

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                sh_trans_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                # cliping the weight norm
                # grads_and_vars = sh_trans_optimizer.compute_gradients(self.l_trans_sh)
                # capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=10, axes=0), gv[1])
                #              for gv in grads_and_vars]
                # sh_trans_optimizer = sh_trans_optimizer.apply_gradients(capped_grads_and_vars)
                sh_trans_train_op = sh_trans_optimizer.minimize(self.l_trans_sh, global_step=global_step, var_list=trans_variables)

                if self.F_init is None:
                    wgan_trans_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate*10)
                    wgan_trans_train_op = wgan_trans_optimizer.minimize(self.l_trans_wgan, global_step=global_step, var_list=trans_variables)
                    disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                    disc_train_op = disc_optimizer.minimize(self.l_disc, var_list=disc_variables)

            sess.run(tf.global_variables_initializer())
            step = 0

            batches_src = uniform_batch_iter([self.vecs_list[0], self.freq_list[0]], batch_size=batch_size, num_epochs=epoch, shuffle=True, full_batch=True, verbose=False)
            batches_tgt = uniform_batch_iter([self.vecs_list[1], self.freq_list[1]], batch_size=batch_size, num_epochs=epoch, shuffle=True, full_batch=True, verbose=False)
            val_acc = 0.0
            cur_lr = lr
            epoch_start_time = time.time()

            for batch_src, batch_tgt in zip(batches_src, batches_tgt):
                x_src_batch = batch_src[0][0]
                x_tgt_batch = batch_tgt[0][0]
                freq_src_batch = batch_src[0][1]
                freq_tgt_batch = batch_tgt[0][1]

                # more advenced lr scheduler
                # T = int(step/batch_per_epoch) + 1
                # cur_lr = lr_schedule_dict[T]    
                '''OR'''
                # if step < total_steps/2:
                #     cur_lr = lr
                # else:
                #     cur_lr = lr/10
                '''OR'''
                # simplest lr scheduler
                # cur_lr = lr

                feed_dict = {
                  self.input_s: x_src_batch,
                  self.input_t: x_tgt_batch,
                  self.dp_keep_prob: dp_keep_prob,
                  self.freq_s: freq_src_batch,
                  self.freq_t: freq_tgt_batch,
                  learning_rate: cur_lr,
                  self.phase: True,
                }
                if self.F_init is not None:
                    feed_dict[self.input_s_init] = self.init_vecs_list[0]
                    feed_dict[self.input_t_init] = self.init_vecs_list[1]

                if self.F_init is None and (not swith_flag): 
                    
                    # update WGAN disc models
                    for _ in range(DISC_ITERS):
                        _, l_disc, l_trans_wgan = sess.run([disc_train_op, self.l_disc, self.l_trans_wgan], feed_dict)
                    # print("l_trans_wgan before wgan_trans_train_op", l_trans_wgan)

                    # update WGAN transfer models
                    # for _ in range(WGAN_TRANS_ITERS):
                    _, step, loss_reconstruct, loss_comb, l_trans_wgan, summary = sess.run([wgan_trans_train_op, global_step, self.reconstruct_loss, self.sinkhorn_distance, self.l_trans_wgan, self.merged_summary], feed_dict)
                    # print("l_trans_wgan after wgan_trans_train_op", l_trans_wgan)

                    # judge convergence, if so, switch training loss
                    # if early_stopper.check_early_stop(loss_comb, int( (step-0.1) / batch_per_epoch)+1) or step > total_steps/2:
                    if step > wgan_epoch * batch_per_epoch:
                        swith_flag = True
                        if not self.use_sinkhorn:
                            logger.info('Epoch, {0}, end up just using WGAN!'.format(int( (step-0.1) / batch_per_epoch)+1))
                            break
                        logger.info('Epoch, {0}, switch to sinkhorn loss!'.format(int( (step-0.1) / batch_per_epoch)+1))
                        self.saver.restore(sess, os.path.join(self.save_path, 'model.ckpt'))
                        early_stopper = EarlyStopper(patience*batch_per_epoch, min_delta)
                        model_checker = ModelChecker()
                        cur_lr = lr
                elif self.use_sinkhorn: # update sinkhorn transfer models
                    for _ in range(TRANS_ITER):
                        _, step, loss_reconstruct, loss_comb, init_align_loss, summary = sess.run([sh_trans_train_op, global_step, self.reconstruct_loss, self.l_trans_sh, self.init_align_loss, self.merged_summary], feed_dict)

                    # handle early stopping
                    # if step % batch_per_epoch == 0:
                    if early_stopper.check_early_stop(loss_comb, int( (step-0.1) / batch_per_epoch)+1):
                        logger.info('Epoch, {0} early stopping!'.format(int( (step-0.1) / batch_per_epoch)+1))
                        print('Epoch, {0} early stopping!'.format(int( (step-0.1) / batch_per_epoch)+1))
                        break
                else:
                    print('ERROR: Invalid config for training!')

                # validation and print
                if step % (50*batch_per_epoch) == 0:
                    validation_start_time = time.time()
                    stdout_buffer = ''
                    self.writer.add_summary(summary, step)
                    stdout_buffer += 'VALIDATION: Epoch:{} '.format(int( (step-0.1) / batch_per_epoch)+1)

                    if DEBUG:
                        for w in ['城市', '小行星', '文学']:
                            w_knn = self.find_k_nearest(w=w, k=5, src2tgt=True)
                            print('word: %s has knn words --- %s' % (w, ' '.join(w_knn)))
                    if validation is not None:
                        self.transfer_tgt_emb(sess=sess)
                        # print('validation[0][:10]:', validation[0][:10])
                        # print('validation[1][:10]:', validation[1][:10])
                        nnr_tgt = self.find_nearest_gpu(validation[0], src2tgt=True, batch_size=batch_size, sess=sess)
                        # print('nnr_tgt[:10]:', nnr_tgt[:10])
                        val_acc = knn_accuracy_from_list(nnr_tgt, validation[1], k=1)
                        stdout_buffer += 'Bilex acc:{0:.4f} '.format(val_acc)
                        val_acc_value = tf.Summary.Value(tag='val_acc', simple_value=val_acc)
                        self.writer.add_summary(tf.Summary(value=[val_acc_value]), step)

                    stdout_buffer += 'Validation eclipsed time \n'.format(time.time()-validation_start_time)
                    sys.stdout.write(stdout_buffer)
                    sys.stdout.flush()

                # check objective to save the best model
                model_checker.record_loss(loss_comb, int( (step-0.1) / batch_per_epoch)+1)
                if step % batch_per_epoch == 0:
                    print('Epoch  {}, time eclipsed {}'.format(int((step-0.1) / batch_per_epoch)+1, time.time()-epoch_start_time))
                    epoch_start_time = time.time()
                    if model_checker.check_for_best(loss_comb, int((step-0.1) / batch_per_epoch)+1):
                        save_path = self.saver.save(sess, os.path.join(self.save_path, 'model.ckpt'))
                        best_loss = model_checker.get_best_loss()
                        print('Saved at combined loss {}'.format(best_loss))
                        if validation is not None and logger is not None:
                            logger.info('Saved at epoch:{0}, step:{1} loss:{2:.4f} Bilingual Induction Accuracy --- {3:.4f}'.format(int( (step-0.1) / batch_per_epoch)+1, step, best_loss, val_acc))
                        cur_lr = lr
                    else:
                        cur_lr = lr*0.95

        self.writer.close()
        return val_acc