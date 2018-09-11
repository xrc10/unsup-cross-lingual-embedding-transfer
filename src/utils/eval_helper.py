import numpy as np
from os import listdir 
from os.path import join, isdir
from collections import defaultdict
from utils.data_helper import load_bi_dict

def knn_accuracy_from_list(pred_list, true_list, k=1):
    true_count = 0.0
    for preds, trues in zip(pred_list, true_list):
        for pred in preds[:k]:
            if pred in trues:
                true_count += 1
                break
    return true_count/len(true_list)

def contain_cand_token(word_list, cand_list):
    for w in word_list:
        if w in cand_list:
            return True
    return False

def read_validation(dict_path, word_list):
    q_words = list()
    true_words = list()

    # read dictionary
    bi_dict = load_bi_dict(dict_path)
    # print('bi_dict', bi_dict)
    # iterate target words
    for q_word in word_list[0]:
        # print(q_word)
        if q_word in bi_dict and contain_cand_token(bi_dict[q_word], word_list[1]):
            q_words.append(q_word)
            true_words.append(bi_dict[q_word])

    return (q_words, true_words)

def find_common_words(words, freqs, qw):
    max_freq = 0
    res = None
    for i,w in enumerate(words):
        if w == qw and freqs[i] > max_freq:
            res = i
            max_freq = freqs[i]
    return res

def knn_accuracy_from_matrix(F_pred, F_gold, k=1, src_words=None, tgt_words=None, verbose=False):
    total = 0.0
    tp = 0.0
    gold_sum = np.sum(F_gold, axis=1)
    # print(np.sum(F_pred, axis=1))
    # print(gold_sum)
    for i in range(F_gold.shape[0]):
        if gold_sum[i] > 0:
            total += 1.0
            if verbose:
                print("Query: ", src_words[i])
                print("Gold:")
                for gold_j in np.nonzero(F_gold[i,:])[0].tolist():
                    print("\t" + tgt_words[gold_j])

                print("System:")
            for j in np.argsort(-F_pred[i,:]).tolist()[:k]:
                if verbose:
                    print("\t" + tgt_words[j])
                if F_gold[i,j] == 1:
                    tp += 1.0
    print("evaluated on %d pairs" % (total))
    return tp/total