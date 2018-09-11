#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
USAGE:
OUTPUT:

"""

import torch
import os
import numpy as np
import random
from os import listdir 
from os.path import join, isdir
from tqdm import tqdm
tqdm.monitor_interval = 0
from numpy import linalg as LA
from collections import defaultdict
import errno
from sklearn import preprocessing
from scipy import stats
from itertools import cycle

def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

def my_scale(X, axis=0):
    '''From scikit-learn preprocessing.scale'''
    Xr = np.asarray(X)
    mean_ = np.mean(X, axis)
    scale_ = np.std(X, axis)
    scale_ = _handle_zeros_in_scale(scale_, copy=False)

    Xr -= mean_
    Xr /= scale_

    return Xr, mean_, scale_

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def file_len(fname):
    with open(fname, errors='surrogateescape') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def load_bi_dict(fname, splitter='\t'):
    bi_dict = defaultdict(list)
    for l in open(fname, errors='surrogateescape'):
        if splitter in l:
            ss = l.strip().split(splitter)
        else:
            ss = l.strip().split(' ')
        bi_dict[ss[0]].append(ss[1])
    return bi_dict

def load_text_vec(fname, splitter=' ', vocab_size=None, top_n=None, norm=None):
    """
    Load dx1 word vecs from word2vec-like format:
    <word1> <dim1> <dim2> ...
    <word2> <dim1> <dim2> ...
    ...
    """
    # word_vecs = defaultdict(list)
    word_vecs = dict()
    words = []
    vecs = []
    with open(fname, "r", errors='surrogateescape') as f:
        if vocab_size is None:     
            vocab_size = file_len(fname)
        layer1_size = None
        vocab_count = 0

        for line in tqdm(f.readlines()[:min(vocab_size,top_n)+1]):
            ss = line.split(' ')
            if len(ss) <= 3:
                continue
            word = ss[0]
            dims = ' '.join(ss[1:]).strip().split(splitter)
            if layer1_size is None:
                layer1_size = len(dims)
                # print dims
                print("reading word2vec at vocab_size:%d, dimension:%d" % (vocab_size, layer1_size))

            vec = np.fromstring(' '.join(dims), dtype='float32', count=layer1_size, sep=' ')
            vecs.append(vec)
            words.append(word)
            vocab_count += 1
            if vocab_count >= vocab_size:
                break
            if top_n is not None:
                if vocab_count >= top_n:
                    break

        vecs = np.asarray(np.stack(vecs, axis=0))
        if norm == 'scale':
            vecs = preprocessing.scale(vecs)
        elif norm == 'l2':
            vecs = preprocessing.normalize(vecs, norm='l2', axis=1)
        elif norm == 'l2+mean_center':
            vecs = preprocessing.normalize(vecs, norm='l2', axis=1)
            vecs = mean_center(vecs)
        elif norm == 'none':
            pass
        else:
            print('ERROR: unrecoginized norm optioin {}'.format(norm))
        # print("vecs.mean(axis=0)", vecs.mean(axis=0))

        scaled_vecs, mean, std = my_scale(vecs)

        # vecs = np.copy(scaled_vecs)
        # vecs = preprocessing.normalize(vecs, norm='l2', axis=1)
        # vecs *= std
        # vecs += mean

        for i, word in enumerate(words):
            word_vecs[word] = vecs[i,:]

        print('vecs.shape:', vecs.shape)
    return vocab_size, word_vecs, words, vecs, scaled_vecs, mean, std, layer1_size

# def cycle_zip(A, B): # return zip of list A,B, which are in different sizes 
#     return zip(A, cycle(B)) if len(A) > len(B) else zip(cycle(A), B)

def downsample_frequent_words(counts, total_count, frequency_threshold=1e-3):
    if total_count > 1: # if inputs are counts
        threshold_count = float(frequency_threshold * total_count)
        probs = (np.sqrt(counts / threshold_count) + 1) * (threshold_count / counts)
        probs = np.maximum(probs, 1.0)    #Zm: Originally maximum, which upsamples rare words
        probs *= counts
        probs /= probs.sum()
    elif total_count <= 1: # inputs are frequency already
        probs = np.power(counts, 0.75)
        probs /= probs.sum()
    return probs

def load_freq(p, top_n=None, splitter=' '):
    """
    Load word frequence from word count
    <word1> <count1> ...
    <word2> <count2> ...
    ...
    return <word:word_count dictionary>, <freq_list>
    """
    if p is None:
        # assume uniform distribution
        return None, np.ones(top_n)/top_n 

        # assume Zipf's law
        # freq = 1/(np.arange(top_n)+1)
        # return None, downsample_frequent_words(freq/np.sum(freq), 0)

    else:
        w_count = defaultdict(list)
        count_list = []
        for l in open(p):
            ss = l.strip().split(splitter)
            w_count[ss[0]].append(float(ss[1]))
            count_list.append(float(ss[1]))
            if len(count_list) >= top_n:
                break
        counts = np.asarray(count_list, dtype=np.float32)
        total_count = counts.sum()
    return w_count, downsample_frequent_words(counts, total_count)

def uniform_batch_iter(xs, batch_size=32, num_epochs=1, shuffle=True, full_batch=True, verbose=False):
    """
    Generates a batch iterator for a dataset. Use uniform sample from data.
    xs: list of numpy array data, must has equal size on axis 0
    batch_size: size of batch
    num_epochs: number of epochs
    shuffle: True will shuffle xs with same random indices, False will not shuffle
    full_batch: True to make use every batch has the same batch_size, False the last batch of each epoch may contain samples less than batch_size
    verbose: True to print epoch and batch size info on the run 
    """

    data_sizes = [x.shape[0] for x in xs]
    assert data_sizes[1:] == data_sizes[:-1] # make sure they are all equal
    data_size = data_sizes[0]
    num_batches_per_epoch = int((data_size-0.1)/batch_size) + 1
    # make data be n times size of batch size
    res_size = batch_size - data_size % batch_size
    if res_size > 0 and full_batch:
        shuffle_indices = np.random.choice(np.arange(data_size), size=res_size, replace=False)
        xs = [np.concatenate((x, x[shuffle_indices]), axis=0) for x in xs]
        data_size = xs[0].shape[0]
    # print(num_epochs*num_batches_per_epoch)
    for epoch in range(num_epochs):
        if shuffle: # Shuffle the data at each epoch
            xs_shuffled = list()
            shuffle_indices = np.random.permutation(np.arange(data_size)) # same shuffle for all data
            for i,x in enumerate(xs):    
                xs_shuffled.append(x[shuffle_indices])
        else:
            xs_shuffled = xs
        if verbose:
            sys.stdout.write("\rIn epoch >> " + str(epoch + 1))
            sys.stdout.flush()
            # print("In epoch >> " + str(epoch + 1), end='', flush=True)
            # print("num batches per epoch is: " + str(num_batches_per_epoch), end='', flush=True)
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            xs_batch = [x[start_index:end_index] for x in xs_shuffled]
            batch = (xs_batch, start_index, end_index, epoch)
            yield batch

def freq_sample_batch_iter(xs, ps, batch_size=32, num_epochs=1, verbose=False):
    """
    Generates a batch iterator for a dataset. Use frequence sample from ps.
    """
    data_size = max([x.shape[0] for x in xs]) # find the max data size
    num_batches_per_epoch = int((data_size-0.1)/batch_size) + 1
    for epoch in range(num_epochs):
        if verbose:
            print("In epoch >> " + str(epoch + 1))
            print("num batches per epoch is: " + str(num_batches_per_epoch))
        for batch_num in range(num_batches_per_epoch):
            xs_batch = []
            for x, p in zip(xs, ps):
                idx = np.random.choice(np.arange(x.shape[0]), size=batch_size, replace=True, p=p)
                xs_batch.append(x[idx,:])
            yield (xs_batch, None, None)


# utility function
def mean_center(matrix, axis=0):
    # print(type(matrix))
    if type(matrix) is np.ndarray:
        avg = np.mean(matrix, axis=axis, keepdims=True)
    else:
        avg = torch.mean(matrix, axis=axis, keepdims=True)
    return matrix - avg

def get_wordvec(filename, top_n=None):
    arr = []
    words = []
    count = 0
    for num, line in enumerate(open(filename)):
        if num == 0 and len(line.split()) < 4:
            continue
        word, vect = line.rstrip().split(' ', 1)
        arr.append(vect.split())
        # assert len(vect.split()) == 300
        words.append(word)
        count += 1
        if top_n is not None and count >= top_n:
            break
    return np.vstack(arr).astype(float), words
    # return mean_center(np.vstack(arr).astype(float), axis=0), words

def save_emb(filename, X, words):
    """
    save vectors X(np.array) and words to file f
    """
    f = open(filename, 'w', errors='surrogateescape')
    num_words = X.shape[0]
    assert num_words == len(words), "saved words is not aligned with saved vectors"
    f.write('{} {}\n'.format(num_words, X.shape[1]))
    for i,word in enumerate(words):
        word_vec_str = ' '.join(map(str, X[i,:].tolist()))
        f.write("{} {}\n".format(word, word_vec_str))

def get_dictionary_index(filename, src_words, tgt_words, limit=None, unique=False):
    src_idx, tgt_idx = [], []
    counter = 0
    for i,l in enumerate(open(filename).readlines()):
        if limit is not None and counter >= limit:
            break
        ss = l.strip().split()
        if ss[0].lower() in src_words and ss[1].lower() in tgt_words:
            if unique and (src_words.index(ss[0].lower()) in src_idx or tgt_words.index(ss[1].lower()) in tgt_idx):
                continue
            src_idx.append(src_words.index(ss[0].lower()))
            tgt_idx.append(tgt_words.index(ss[1].lower()))
            counter += 1
    return src_idx, tgt_idx

def get_dictionary_matrix(filename, src_words, tgt_words, limit=None):
    """
    Suppose the dictionary has line format: <src_word> <tgt_word>
    """
    F = np.zeros((len(src_words),len(tgt_words)))
    src_idx, tgt_idx = get_dictionary_index(filename, src_words, tgt_words, limit=limit)
    F[np.asarray(src_idx), np.asarray(tgt_idx)] = 1
    return F

def get_data(src, tgt, top_n=200000):
    src_arr, src_words = get_wordvec(src, top_n)
    tgt_arr, tgt_words = get_wordvec(tgt, top_n)
    return src_arr, src_words, tgt_arr, tgt_words

def sparsify_mat(K, nn):
    ret_K = np.zeros(K.shape)
    for i in range(K.shape[0]):
        index = np.argsort(K[i, :])[-nn:]
        ret_K[i, index] = K[i, index]
    return ret_K

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def sym_sparsify_mat(K, nn):
        K_sp = sparsify_mat(K, nn)
        K_sp = (K_sp + K_sp.T) / 2  # in case of non-positive semi-definite
        return K_sp

def get_adj(basedir, src, tgt, nn, logger, normalize=True):
    logger.info('Loading data...')
    src_arr, tgt_arr = get_data(basedir, src, tgt)
    logger.info('Loading data finished')
    if normalize:
        src_arr = src_arr / np.linalg.norm(src_arr, ord=2, axis=1, keepdims=True)
        tgt_arr = tgt_arr / np.linalg.norm(tgt_arr, ord=2, axis=1, keepdims=True)
    src_adj = sym_sparsify_mat(src_arr.dot(src_arr.T), nn)
    tgt_adj = sym_sparsify_mat(tgt_arr.dot(tgt_arr.T), nn)
    logger.info('Sparsification finished')
    # print(type(src_adj), type(tgt_adj))
    return torch.from_numpy(src_adj.astype(float)), torch.from_numpy(tgt_adj.astype(float))

if __name__ == '__main__':
    pass
    basedir = '../data/'
    src = 'es'
    tgt = 'en'
    nn = 5

    src_arr, tgt_arr = get_data(basedir, src, tgt)

    src_adj = sym_sparsify_mat(src_arr.dot(src_arr.T), nn)
    tgt_adj = sym_sparsify_mat(tgt_arr.dot(tgt_arr.T), nn)
    print(src_adj.shape)
    print(tgt_adj.shape)
    # print src_adj[0]
