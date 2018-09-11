from __future__ import print_function
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import argparse
import json
# from model.data_helper import load_text_vec, load_bi_dict
# from eval_knn_acc import load_freq
# from tqdm import tqdm
import operator
sys.path.append('/usr1/home/ruochenx/research/cross_emb/Unsup_solver/env/lib/python3.4/site-packages')
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.sparse import coo_matrix
# import pinyi

def plot_emb(X, Y, fig_path, X_words=None, Y_words=None):
    fig, ax = plt.subplots()
    assert X.shape[1] == Y.shape[1], "ERROR(plot_emb): X, Y dimension not match"
    # assert X.shape[0] ==  len(X_words) and Y.shape[0] == len(Y_words), "ERROR(plot_emb): X, Y data count not match {} {} {} {}".format(X.shape[0], Y.shape[0], len(X_words), len(Y_words))

    Z_2d = np.concatenate((X, Y), axis=0)
    if X.shape[1] > 2:
        Z_2d = my_tsne(Z_2d)

    colors = ['red'] * X.shape[0] + ['blue'] * Y.shape[0]
    ax.scatter(Z_2d[:,0], Z_2d[:,1], color=colors, s=1)
    if X_words is not None and Y_words is not None:
        for i, txt in enumerate(X_words + Y_words):
            ax.annotate(txt, xy=(Z_2d[i,0],Z_2d[i,1]), fontsize=1)
    fig.savefig(fig_path)
    plt.close(fig)


def my_tsne(X):
    """
    convert X into 2-dimension vectors
    """
    print("TSNE for data of size:", X.shape)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    return X_embedded

def filter_word_vecs(bi_dict, src_word_vecs, tgt_word_vecs, total_count=2000):
    words = []
    vecs = []
    colors = []
    count = 0
    keep_list = []
    for tgt_w in bi_dict:
        for src_w in bi_dict[tgt_w]:
            if tgt_w in tgt_word_vecs and src_w in src_word_vecs and not tgt_w in words and not src_w in words:
                if (src_w, tgt_w) in keep_list:
                    continue
                keep_list.append((src_w, tgt_w))
                words.append(src_w)
                vecs.append(src_word_vecs[src_w])
                colors.append('red')
                words.append(pinyin.get(tgt_w, format="numerical"))
                vecs.append(tgt_word_vecs[tgt_w])
                colors.append('blue')
                count += 1
                if count > total_count:
                    return np.stack(vecs, axis=0), words, colors

    return np.stack(vecs, axis=0), words, colors

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

def plot_align_matrix(F, file):
    # sns_plot = sns.heatmap(F, center=0.5)
    # fig = sns_plot.get_figure()
    ax = plot_coo_matrix(F)
    ax.figure.savefig(file)

def main():
    parser = argparse.ArgumentParser(description='visualize the shared embedding space')
    parser.add_argument('-src_emb_path', help='path to the source embedding model')
    parser.add_argument('-src_freq_path', help='path to the source word frequency file')
    parser.add_argument('-tgt_emb_path', help='path to the target embedding model')
    parser.add_argument('-tgt_freq_path', help='path to the target word frequency file')
    parser.add_argument('-dictionary', help='path to the gold dictionary')
    parser.add_argument('-fig_out', help='path to the output figure')
    args = parser.parse_args()
    # read dictionary
    bi_dict = load_bi_dict(args.dictionary)

    # read embeddings
    word_vecs_list = []
    words_list = []
    vecs_list = []
    word_freq_list = []
    for p in [(args.src_emb_path, args.src_freq_path), (args.tgt_emb_path, args.tgt_freq_path)]:
        _, word_vecs, words, vecs, emb_dim = load_text_vec(p[0])
        print('Loaded embeddings of %d words at dimension %d' % (len(word_vecs), emb_dim))
        word_freq = load_freq(p[1])

        word_vecs_list.append(word_vecs)
        words_list.append(words)
        vecs_list.append(vecs)
        word_freq_list.append(word_freq)

    vecs, words, colors = filter_word_vecs(bi_dict, word_vecs_list[0], word_vecs_list[1])
    plot_emb(bi_dict, vecs, words, colors, args.fig_out, total_count=100)

if __name__ == '__main__':
    main()