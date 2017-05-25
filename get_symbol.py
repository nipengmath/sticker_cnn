#coding:utf-8

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")

import gpu_mxnet as mx
import numpy as np

import json


def get_text_cnn_symbol(location):
    with open(location) as f:
        text = f.read()
    x = json.loads(text)
    for n in x['nodes']:
        if n['op'] == 'Reshape':
            print n['name']
            n['param']['target_shape'] = "(1,240)"
    text = json.dumps(x)
    #sm = mx.symbol.load(location)
    sm = mx.symbol.load_json(text)
    arg_names = sm.list_arguments()
    for arg in arg_names:
        print arg
    #print arg_names
    #print sm.get_internals()

    #print dir(sm)
    #print sm.debug_str()


  


get_text_cnn_symbol('cnn-symbol.json')
