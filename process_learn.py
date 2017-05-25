#coding:utf-8

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")

import gpu_mxnet as mx

#import mxnet as mx
import numpy as np
import data_store
from text_cnn import get_text_cnn_symbol, get_text_cnn_model, learn

def process_learn(location_vec, location_ins, model_root, gpu, prop={},epoch=20):
    sentence_size = prop.get('sentence_size', 25)
    batch_size = prop.get('batch_size', 128)
    num_label = prop.get('num_label', 2)
    filter_list = prop.get('filter_list', [1, 2, 3, 4])
    num_filter = prop.get('filter_num', 60)
    dropout = prop.get('dorpout', 0.5)
    w2v = data_store.get_w2v(location_vec)
    print 'w2v', len(w2v)
    x, y, vocab, vocab_inv = data_store.get_data_w2v(location_ins, w2v, sentence_size)
    vocab_size = len(vocab)
    print 'data'
    print x.shape
    print y.shape
    cc = len(x) / 10.0
    cc = int(cc) * 1
    x_train, x_dev = x[:-cc], x[-cc:]
    y_train, y_dev = y[:-cc], y[-cc:]
    print 'Train/Dev split: %d/%d' % (len(y_train), len(y_dev))
    print 'train shape:', x_train.shape
    print 'dev shape:', x_dev.shape
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_dev = np.reshape(x_dev, (x_dev.shape[0], 1, x_dev.shape[1], x_dev.shape[2]))
    vec_size = x_train.shape[-1]
    sentence_size = x_train.shape[2]
    cnn = get_text_cnn_symbol(sentence_size, vec_size, batch_size, dropout=dropout)
    ctx = mx.gpu(gpu)
    cm = get_text_cnn_model(ctx, cnn, sentence_size, vec_size, batch_size)
    print 'batch_size', batch_size
    print 'sentence_size', sentence_size
    print 'vec_size', vec_size
    learn(cm, x_train, y_train, x_dev, y_dev, batch_size, epoch=epoch,root=model_root)


