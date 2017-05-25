#coding:utf-8

#import sys
#sys.path.insert(0,"/opt/tiger/text_lib/")

import json

import mxnet as mx

#import mxnet as mx
import numpy as np
from text_iterator import get_vocab, TextIterator
from text_cnn import get_text_cnn_symbol,get_text_cnn_model
from text_cnn_fit import fit


def process_fit(location_vec, location_ins, location_test, model_dir, gpu, prop={}, epoch=20):
    sentence_size = prop.get('sentence_size', 25)
    batch_size = prop.get('batch_size', 128)
    num_label = prop.get('num_label', 2)
    filter_list = prop.get('filter_list', [1, 2, 3, 4])
    num_filter = prop.get('num_filter', 60)
    dropout = prop.get('dorpout', 0.5)

    id2v, w2id = get_vocab(location_vec)
    print 'w2id',len(w2id)

    text_iter = TextIterator(w2id, location_ins, sentence_size)
    test_iter = TextIterator(w2id, location_test, sentence_size)

    print 'data',text_iter.cnt, test_iter.cnt

    vocab_size, vec_size = id2v.shape

    cnn = get_text_cnn_symbol(vocab_size, sentence_size, vec_size, batch_size,\
            num_label=num_label,filter_list=filter_list,num_filter=num_filter,dropout=dropout)

    if int(gpu) >= 0:
        ctx = mx.gpu(gpu)
    else:
        ctx = mx.cpu(0)

    cm = get_text_cnn_model(ctx,cnn,id2v, sentence_size,batch_size)
    print 'batch_size',batch_size
    print 'sentence_size',sentence_size
    print 'vec_size',vec_size
    location_size = model_dir+'size.txt'
    s = {}
    s['batch_size'] = batch_size
    s['sentence_size'] = sentence_size
    s['vocab_size'] = vocab_size
    s['vec_size'] = vec_size
    s['location_vec'] = location_vec
    s['epoch_num'] = epoch
    s['num_label'] = num_label
    s['filter_list'] = filter_list
    s['num_filter'] = num_filter
    s['dropout'] = dropout

    line = json.dumps(s)
    with open(location_size,mode='w') as f:
        f.write(line)

    fit(cm,text_iter,test_iter,batch_size,epoch=epoch,root=model_dir)
    # text_iter.close()
    # test_iter.close()


if __name__ == '__main__':
    location_vec = 'w2v.m'
    location_ins = 'location.train'
    location_test = 'location.test'
    model_dir = 'model/'
    gpu = -1
    prop = {}
    epoch = 5
    process_fit(location_vec, location_ins, location_test, model_dir, gpu, prop, epoch)
