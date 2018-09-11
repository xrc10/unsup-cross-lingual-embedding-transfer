#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

import os, sys
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from sklearn import preprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_helper import load_text_vec, load_freq, save_emb, uniform_batch_iter, freq_sample_batch_iter, mkdir_p
from utils.eval_helper import knn_accuracy_from_list, find_common_words

from model.distance import *
from model.common_layers import *

USE_NORM_LAYER = False
USE_EUC_DIST = False
TIED_WEIGHTS = True
ORTH_INIT = True

################################################################################
### The parent class for transfer-based embedding model

class CycleEmb:
    def __init__(self, trans_hidden_dim=[], disc_hidden_dim=[50], recon_loss_weight=[1,1], trans_activation=False, cyc_loss=True, use_sinkhorn=True, use_BN=True, norm='scale', save_path=None):
        self.trans_hidden_dim = trans_hidden_dim # hidden dimension of transfer network
        self.disc_hidden_dim = disc_hidden_dim
        self.recon_loss_weight = recon_loss_weight # relative weight of reconstruction loss
        self.trans_activation = trans_activation
        self.cyc_loss = cyc_loss
        self.use_sinkhorn = use_sinkhorn
        self.use_BN = use_BN
        self.saver = None
        self.save_path = save_path
        self.norm = norm
        mkdir_p(save_path)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=gpu_options)

    def set_highway_layer(self, hw_hidden_dim=[50]):
        self.highway_nonlinear = True
        self.hw_hidden_dim = hw_hidden_dim # hidden dimension of highway network

    def load_embs(self, emb_p_list, freq_p_list, top_n_vec=200000):
        """ 
        load two sets of monolingual embeddings 
        p_list: [path_to_embedding1, path_to_embedding2] 
        """
        self.word_vecs_list = []
        self.words_list = []
        self.ori_vecs_list = [] # store the vectors after preprocessing
        self.vecs_list = [] # store the vectors after preprocessing and scaling
        self.mean_list = [] # store the info of mean of scale
        self.std_list = [] # store the info for std of scale
        self.word_count_list = []
        self.freq_list = []
        self.emb_dim_list = []
        for emb_p, freq_p in zip(emb_p_list, freq_p_list):
            _, word_vecs, words, vecs, scaled_vecs, mean, std, emb_dim = load_text_vec(emb_p, top_n=top_n_vec, norm=self.norm)
            print('Loaded embeddings of %d words at dimension %d' % (len(word_vecs), emb_dim))
            self.ori_vecs_list.append(vecs)
            self.word_vecs_list.append(word_vecs)
            self.words_list.append(words)
            self.emb_dim_list.append(emb_dim)
            
            word_count_d, word_freqs = load_freq(p=freq_p, top_n=len(word_vecs))
            print('Loaded frequency of %d senses' % (len(word_freqs)))
            self.freq_list.append(word_freqs)
            self.word_count_list.append(word_count_d)
            
            self.vecs_list.append(scaled_vecs)
            self.mean_list.append(mean)
            self.std_list.append(std)
            
    def build_input_layers(self):
        # input embeddings
        self.input_s = tf.placeholder(tf.float32, shape = [None, self.emb_dim_list[0]], name='souce_emb_input')
        self.input_t = tf.placeholder(tf.float32, shape = [None, self.emb_dim_list[1]], name='target_emb_input')

        self.freq_s = tf.placeholder(tf.float32, shape = [None,], name='souce_freq_input')
        self.freq_t = tf.placeholder(tf.float32, shape = [None,], name='target_freq_input')

        # self.freq_s = tf.ones([tf.shape(self.input_s)[0], 1], tf.float32)/tf.cast(tf.shape(self.input_s)[0], dtype=tf.float32)
        # self.freq_t = tf.ones([tf.shape(self.input_t)[0], 1], tf.float32)/tf.cast(tf.shape(self.input_t)[0], dtype=tf.float32)

        self.dp_keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.phase = tf.placeholder(tf.bool, name='phase') # True if in training phrase, False elsewise
        if self.trans_activation == 'selu':
            self.input_s = tf.contrib.nn.alpha_dropout(self.input_s, self.dp_keep_prob)
            self.input_t = tf.contrib.nn.alpha_dropout(self.input_t, self.dp_keep_prob)
        else:
            self.input_s = tf.nn.dropout(self.input_s, self.dp_keep_prob)
            self.input_t = tf.nn.dropout(self.input_t, self.dp_keep_prob)
        if self.F_init is not None:
            self.input_s_init = tf.placeholder(tf.float32, shape = [None, self.emb_dim_list[0]], name='souce_emb_input_init')
            self.input_t_init = tf.placeholder(tf.float32, shape = [None, self.emb_dim_list[1]], name='target_emb_input_init')

    def build_transfer_layers(self, multiple_noise_std=None, tied_weights=TIED_WEIGHTS):    
        # embedding transfer layers
        # batch transfer
        trans_s, self.Ws_s2t, _ = dense_layers(input_layer=self.input_s, input_dim=self.emb_dim_list[0], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[1], name='fc-src2tgt', activation=self.trans_activation, bias=False, orth_init=ORTH_INIT, add_multiple_noise=multiple_noise_std, BN=self.use_BN, BN_phase=self.phase, BN_reuse=False)
        self.trans_s = l2_norm(trans_s) if USE_NORM_LAYER else trans_s
        
        if tied_weights:
            self.Ws_t2s = [tf.transpose(W) for W in reversed(self.Ws_s2t)]
            trans_t, _, _ = dense_layers(input_layer=self.input_t, input_dim=self.emb_dim_list[1], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[0], Ws=self.Ws_t2s, name='fc-tgt2src', activation=self.trans_activation, bias=False, orth_init=ORTH_INIT, add_multiple_noise=multiple_noise_std, BN=self.use_BN, BN_phase=self.phase, BN_reuse=False)
        else:
            trans_t, self.Ws_t2s, _ = dense_layers(input_layer=self.input_t, input_dim=self.emb_dim_list[1], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[0], name='fc-tgt2src', activation=self.trans_activation, bias=False, orth_init=ORTH_INIT, add_multiple_noise=multiple_noise_std, BN=self.use_BN, BN_phase=self.phase, BN_reuse=False)
        self.trans_t = l2_norm(trans_t) if USE_NORM_LAYER else trans_t

        if self.F_init is not None:
        # all transfer(to use the initial dictionary)
            trans_s_init, _, _ = dense_layers(input_layer=self.input_s_init, input_dim=self.emb_dim_list[0], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[1], Ws=self.Ws_s2t, name='fc-src2tgt', activation=self.trans_activation, bias=False, orth_init=ORTH_INIT, BN=self.use_BN, BN_phase=self.phase, BN_reuse=True)
            self.trans_s_init = l2_norm(trans_s_init) if USE_NORM_LAYER else trans_s_init

            trans_t_init, _, _ = dense_layers(input_layer=self.input_t_init, input_dim=self.emb_dim_list[1], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[0], Ws=self.Ws_t2s, name='fc-tgt2src', activation=self.trans_activation, bias=False, orth_init=ORTH_INIT, BN=self.use_BN, BN_phase=self.phase, BN_reuse=True)
            self.trans_t_init = l2_norm(trans_t_init) if USE_NORM_LAYER else trans_t_init

        # batch transfer back to complete the cycle
        trans_back_s, _, _ = dense_layers(input_layer=self.trans_s, input_dim=self.emb_dim_list[1], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[0], Ws=self.Ws_t2s, name='fc-tgt2src', activation=self.trans_activation, bias=False, add_multiple_noise=multiple_noise_std, BN=self.use_BN, BN_phase=self.phase, BN_reuse=True)
        self.trans_back_s = l2_norm(trans_back_s) if USE_NORM_LAYER else trans_back_s

        trans_back_t, _, _ = dense_layers(input_layer=self.trans_t, input_dim=self.emb_dim_list[0], hidden_dim=self.trans_hidden_dim, output_dim=self.emb_dim_list[1], Ws=self.Ws_s2t, name='fc-src2tgt', activation=self.trans_activation, bias=False, add_multiple_noise=multiple_noise_std, BN=self.use_BN, BN_phase=self.phase, BN_reuse=True)
        self.trans_back_t = l2_norm(trans_back_t) if USE_NORM_LAYER else trans_back_t

    def build_recontruction_layers(self):
        if self.cyc_loss:
            if USE_EUC_DIST:
                self.reconstruct_loss = euclidean_distance(self.input_s, self.trans_back_s) / (tf.cast(tf.shape(self.input_s)[0], tf.float32)) + euclidean_distance(self.input_t, self.trans_back_t) / (tf.cast(tf.shape(self.input_t)[0], tf.float32))
            else:
                self.reconstruct_loss = 2 - (cosine_similarity(self.input_s, self.trans_back_s)) / (tf.cast(tf.shape(self.input_s)[0], tf.float32)) - (cosine_similarity(self.input_t, self.trans_back_t)) / (tf.cast(tf.shape(self.input_t)[0], tf.float32))
        else:
            if USE_EUC_DIST:
                self.reconstruct_loss = euclidean_distance(self.input_t, self.trans_back_t) / (tf.cast(tf.shape(self.input_t)[0], tf.float32))
            else:
                self.reconstruct_loss = 1 - (cosine_similarity(self.input_t, self.trans_back_t)) / (tf.cast(tf.shape(self.input_t)[0], tf.float32))

    def build_nearest_neighbor(self):
        # compute nearest neighbor given two input embeddings
        # assuming they are both l2-normalized
        self.nn_query = tf.placeholder(tf.float32, shape = [None, self.emb_dim_list[0]], name='nn_query_emb')
        self.nn_search = tf.placeholder(tf.float32, shape = [None, self.emb_dim_list[0]], name='nn_search_emb')
        self.q2s_sim = tf.matmul(self.nn_query, self.nn_search, transpose_b=True)
        self.nnr_idx = tf.argmax(self.q2s_sim, axis=1)

    def transfer_tgt_emb(self, batch_size=128, sess=None):
        """ 
        transfer the source/target embeddings to the transferred space
        """
        self.trans_vecs = [np.zeros(self.vecs_list[0].shape), np.zeros(self.vecs_list[1].shape)]
        if sess is None:
            sess = tf.Session(config=self.session_conf)
            if self.saver is None:
                self.saver = tf.train.Saver(tf.global_variables())
            self.saver.restore(sess, os.path.join(self.save_path,'model.ckpt'))
        for i in [0, 1]:
            batches_tst = uniform_batch_iter([self.vecs_list[i]], batch_size=batch_size, num_epochs=1, shuffle=False, full_batch=False)
            for batch in batches_tst:
                start_idx = batch[1]
                end_idx = batch[2]
                x_batch = batch[0][0]
                input_batch = self.input_s if i == 0 else self.input_t
                output_batch = self.trans_s if i == 0 else self.trans_t
                feed_dict = {
                  input_batch: x_batch,
                  self.phase: False,
                }
                trans_vecs_batch = sess.run([output_batch], feed_dict)[0]
                self.trans_vecs[i][start_idx:end_idx,:] = trans_vecs_batch
            # print("self.trans_vecs[i].shape", self.trans_vecs[i].shape)
            self.trans_vecs[i] = preprocessing.normalize(self.trans_vecs[i], norm='l2', axis=1)
            self.trans_vecs[i] *= self.std_list[1-i]
            self.trans_vecs[i] += self.mean_list[1-i]

    def save_emb(self, p):
        """ 
        save the (transferred) second set of embeddings to text file
        """
        print('Saving the source embedding...')
        save_emb(os.path.join(p, 'src.emb.txt'), self.ori_vecs_list[0], self.words_list[0])
        print('Saving the target embedding...')
        save_emb(os.path.join(p, 'tgt.emb.txt'), self.ori_vecs_list[1], self.words_list[1])

        print('Saving the transferred source embedding...')
        save_emb(os.path.join(p, 'src.trans.emb.txt'), self.trans_vecs[0], self.words_list[0])        
        print('Saving the transferred target embedding...')
        save_emb(os.path.join(p, 'tgt.trans.emb.txt'), self.trans_vecs[1], self.words_list[1])

    def find_k_nearest(self, w, k=5, src2tgt=True):
        """
        find the k nearest embedding cross-lingual
        src2tgt decide the direction of transfer
        """
        if src2tgt and w in self.word_vecs_list[0]:
            q_word_vec = self.word_vecs_list[0][w]
            search_vecs = self.trans_vecs[1]
            search_words = self.words_list[1]

        elif (not src2tgt) and (w in self.word_vecs_list[1]):
            w_id = self.words_list[1].index(w)
            q_word_vec = self.trans_vecs[1][w_id,:]
            search_vecs = self.ori_vecs_list[0]
            search_words = self.words_list[0]
        else:
            print('%s not in embedding', w)
            return 0
        scores = np.dot(search_vecs, q_word_vec.transpose())
        ind = [(i[0], i[1]) for i in sorted(enumerate(list(scores)), key=lambda x:x[1], reverse=True)]
        # print('Max scores:', [t[1] for t in ind[:k]])
        return [search_words[t[0]] for t in ind[:k]]

    def find_nearest_gpu(self, ws, src2tgt=True, batch_size=128, sess=None):
        """
        find the k nearest embedding cross-lingual
        for all words(src or tgt)
        src2tgt decide the direction of transfer
        """
        if src2tgt:
            # w_ids = [self.words_list[0].index(w) for w in ws]
            w_ids = [find_common_words(self.words_list[0], self.freq_list[0], w) for w in ws]
            q_word_vecs = self.ori_vecs_list[0][w_ids,:]
            search_vecs = self.trans_vecs[1]
            search_words = self.words_list[1]

        elif (not src2tgt):
            # w_ids = [self.words_list[1].index(w) for w in ws]
            w_ids = [find_common_words(self.words_list[1], self.freq_list[1], w) for w in ws]
            q_word_vecs = self.trans_vecs[1][w_ids,:]
            search_vecs = self.ori_vecs_list[0]
            search_words = self.words_list[0]

        if sess is None:
            sess = tf.Session(config=self.session_conf)
            self.saver.restore(sess, os.path.join(self.save_path, 'model.ckpt'))

        nnr_idx = np.zeros([q_word_vecs.shape[0]])
        batches = uniform_batch_iter([q_word_vecs], batch_size=batch_size, num_epochs=1, shuffle=False, full_batch=False)
        for batch in batches:
            start_idx = batch[1]
            end_idx = batch[2]
            query_vecs_batch = batch[0][0]
            feed_dict = {
                  self.nn_query: query_vecs_batch,
                  self.nn_search: search_vecs,
                  self.phase: False,
                }
            nnr_idx_batch = sess.run([self.nnr_idx], feed_dict)
            nnr_idx[start_idx:end_idx] = nnr_idx_batch[0]

        return [[search_words[t]] for t in map(int, nnr_idx.tolist())]