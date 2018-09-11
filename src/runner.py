import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import argparse
import json
import shutil
import logging

import tensorflow as tf

from model.cycle_align_emb import CycleAlignEmb
from utils.eval_helper import read_validation
from utils.data_helper import get_dictionary_matrix, get_data, mkdir_p
from config import config_helper

def get_logger(p):
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler(p)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train bilingual word embedding from monolingual corpus')
    parser.add_argument('--config_path',
        help='path to configuration file', default='en-zh')
    parser.add_argument('--F_file',
        type=str, default=None, help='whether load F from a (seed) dictionary file')
    parser.add_argument('--F_validation',
        help='validation dictionary', default=None)
    parser.add_argument('--seed_size',
        type=int, default=50, help='the size of seed dictionary')
    parser.add_argument('--src',
        type=str, default='es', help='source language')
    parser.add_argument('--src_vec',
        type=str, default='es_vec', help='name of source vector file')
    parser.add_argument('--src_freq',
        type=str, default=None, help='name of source frequency file')
    parser.add_argument('--tgt',
        type=str, default='en', help='target language')
    parser.add_argument('--tgt_vec',
        type=str, default='en_vec', help='name of target vector file')
    parser.add_argument('--tgt_freq',
        type=str, default=None, help='name of target frequency file')
    parser.add_argument('--train_max_size', type=int, default=20000,
        help='select top n vectors from input files for training')
    parser.add_argument('--trans_max_size', type=int, default=200000,
        help='select top n vectors from input files for transfer')
    parser.add_argument('--save', default=None,
        help='path to save the transferred the embedding and learned model')
    parser.add_argument('--train', type=int, default=1,
        help='True if train model from scratch, False will load the model from file')
    parser.add_argument('--log',
        help='log to record learning curve', default=None)
    args = parser.parse_args()

    # get config
    configs, rel_paths = config_helper.get_train_configs(args.config_path)
    mkdir_p(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt)))
    logger = get_logger(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt), 'logger.txt'))
    # iterate all possible config combinations
    for config, rel_path in zip(configs, rel_paths):
        # print('config', config)
        # print('rel_path', rel_path)
        # mkdir and create logger
        mkdir_p(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt), rel_path, 'graph'))

        config['pretrain_emb_paths'] = [args.src_vec, args.tgt_vec]
        config['vocab_freq_paths'] = [args.src_freq, args.tgt_freq]
        if 'w_gan' not in config:
            config['w_gan'] = False
        if 'use_sinkhorn' not in config:
            config['use_sinkhorn'] = True

        if args.F_file is not None:
            Y, Y_words, X, X_words = get_data(args.src_vec, args.tgt_vec, top_n=args.train_max_size)
            F_data = np.transpose(get_dictionary_matrix(args.F_file, Y_words, X_words, limit=args.seed_size))
            logger.info('Initialize with {} dictionary'.format(np.count_nonzero(F_data)))
        else:
            F_data = None

        print('Initialize Model...')
        logger.info('Initialize Model...')
        save_path = os.path.abspath(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt), rel_path + '/'))

        model = CycleAlignEmb(
            trans_hidden_dim=config['trans_hidden_dim'],
            disc_hidden_dim=config['disc_hidden_dim'],
            recon_loss_weight=config['recon_loss_weight'],
            constraint_loss_weight=config['constraint_loss_weight'],
            init_align_loss_weight=config['init_align_loss_weight'],
            trans_activation=config['trans_activation'],
            cyc_loss=config['cyc_loss'],
            use_sinkhorn=config['use_sinkhorn'],
            use_BN=config['use_BN'], norm=config['norm'],
            save_path=save_path, F_init=F_data
        )

        json.dump(config, open(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt), rel_path, 'config.json'), 'w'))
        json.dump(vars(args), open(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt), rel_path, 'args.json'), 'w'))

        dp_keep_prob = getattr(config, 'dp_keep_prob', 1.0)

        if args.train:
            print('Loading {} embeddings for training'.format(args.train_max_size))
            logger.info('Loading {} embeddings for training'.format(args.train_max_size))
            model.load_embs(config['pretrain_emb_paths'], config['vocab_freq_paths'], top_n_vec=args.train_max_size)
        else:
            print('Loading {} embeddings for transfer'.format(args.trans_max_size))
            logger.info('Reading {} embeddings for transfer'.format(args.trans_max_size))
            model.load_embs(config['pretrain_emb_paths'], config['vocab_freq_paths'], top_n_vec=args.trans_max_size)

        print('Building Model...')
        logger.info('Building Model...')
        model.build_model()

        if args.F_validation is not None:
            model.build_nearest_neighbor()
            validation = read_validation(args.F_validation, model.words_list)
            print('Reading Validation data... size:', len(validation[0]))
            logger.info('Reading Validation data... size: {}'.format(len(validation[0])))
            assert len(validation[0]) > 0, 'validation is empty'
        else:
            validation = None
        # print(validation)

        if args.train:
            print('Start Training...')
            logger.info('Start Training...')
            val_acc = model.train(
                batch_size=config['batch_size'], lr=config['lr'],
                epoch=config['epoch'], wgan_epoch=config['wgan_epoch'],
                dp_keep_prob=dp_keep_prob, validation=validation, logger=logger
            )
            logger.info('{} val acc: {}'.format(rel_path, val_acc))

            print('Loading {} embeddings for transfer'.format(args.trans_max_size))
            logger.info('Reading {} embeddings for transfer'.format(args.trans_max_size))
            model.load_embs(config['pretrain_emb_paths'], config['vocab_freq_paths'], top_n_vec=args.trans_max_size)

        print('Transferring embedding...')
        logger.info('Transferring embedding...')
        model.transfer_tgt_emb(batch_size=config['batch_size'])

        print('Saving source/transferred target embeddings...')
        logger.info('Saving source/transferred target embeddings...')
        model.save_emb(os.path.join(args.save, '{}-{}'.format(args.src, args.tgt), rel_path))

        tf.reset_default_graph()
