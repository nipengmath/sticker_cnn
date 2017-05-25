#coding:utf-8

import sys
#sys.path.insert(0,"/opt/tiger/text_lib/")

import mxnet as mx

import os
#import mxnet as mx
import numpy as np
import time
import math
import json
from collections import namedtuple
import data_store
from evaluate import evaluate
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logs = sys.stderr
CNNModel = namedtuple('CNNModel', ['cnn_exec', 'symbol', 'data1', 'data2', 'label', 'param_blocks'])


def get_text_cnn_symbol(vocab_size, sentence_size, vec_size, batch_size,
                        num_label=2, filter_list=[1, 2, 3, 4], num_filter=60, dropout=0.0):
    input_x1 = mx.sym.Variable('data1')  # 文本特征，batch_size, sentence_size
    input_x2 = mx.sym.Variable('data2')  # 图片特征，batch_size,
    input_y = mx.sym.Variable('softmax_label')

    ## >> 文本特征做cnn
    embed_x = mx.sym.Embedding(data=input_x1, input_dim=vocab_size, output_dim=vec_size, name="embedding")

    conv_input = mx.sym.Reshape(data=embed_x, target_shape=(batch_size, 1, sentence_size, vec_size))
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, vec_size), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1, 1))
        pooled_outputs.append(pooli)

    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(dim=1, *pooled_outputs)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))
    ## << 文本特征做cnn

    ## >> 图片特征做cnn
    # lenet
    embed_x2 = mx.sym.Embedding(data=input_x2, input_dim=vocab_size, output_dim=vec_size, name="embedding2")

    conv_input2 = mx.sym.Reshape(data=embed_x2, target_shape=(batch_size, 1, sentence_size, vec_size))
    conv1 = mx.symbol.Convolution(data=conv_input2, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=total_filters)
    ## << 图片特征做cnn

    ## merge
    feature_output = [h_pool, fc2]
    concat = mx.sym.Concat(dim=1, *feature_output)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters*2))

    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight,
                               bias=cls_bias, num_hidden=num_label)
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y,
                              name='softmax')
    return sm

def load_text_cnn_symbol(location, batch_size=128):
    with open(location) as f:
        text = f.read()
    x = json.loads(text)
    # 改变预测时的batch_size
    for n in x['nodes']:
        if n['op'] == 'Reshape':
            shape = eval(n['param']['target_shape'])
            shape = (batch_size, ) + shape[1:]
            n['param']['target_shape'] = '(%s)' % ','.join([str(s) for s in shape])
    text = json.dumps(x)
    #sm = mx.symbol.load(location)
    sm = mx.symbol.load_json(text)
    return sm

def get_text_cnn_model(ctx, cnn, embedding, sentence_size, batch_size, initializer=mx.initializer.Uniform(0.1)):
    arg_names = cnn.list_arguments()
    input_shapes = {}
    input_shapes['data1'] = (batch_size, sentence_size)
    input_shapes['data2'] = (batch_size, sentence_size)
    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    print zip(arg_names, arg_shape)
    print zip(cnn.list_outputs(), out_shape)
    arg_arrays = [ mx.nd.zeros(s, ctx) for s in arg_shape ]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ('softmax_label', 'data1', 'data2', 'embedding_weight', 'embedding2_weight'):
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')
    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if name in ('softmax_label', 'data1', 'data2'):
            continue
        if name == 'embedding_weight':
            arg_dict[name][:] = embedding
            continue
        if name == 'embedding2_weight':
            arg_dict[name][:] = embedding
            continue
        initializer(name, arg_dict[name])
        param_blocks.append((i, arg_dict[name], args_grad[name], name))

    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))
    data1 = cnn_exec.arg_dict['data1']
    data2 = cnn_exec.arg_dict['data2']
    label = cnn_exec.arg_dict['softmax_label']
    return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data1=data1, data2=data2, label=label, param_blocks=param_blocks)
