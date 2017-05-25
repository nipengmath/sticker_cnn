#coding:utf-8

import sys
sys.path.insert(0,"/opt/tiger/text_lib/")

import gpu_mxnet as mx

#import mxnet as mx
import numpy as np

from text_cnn import get_text_cnn_symbol,CNNModel

from data_store import get_w2v

def pad_sentences(sentences, padding_word="</s>",sentence_size=25):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max(len(x) for x in sentences)
    sequence_length = sentence_size
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


class TextCNNScorer(object):
    
    def __init__(self,sentence_size,vec_size,
                    location_vec,location_model,location_symbol):
        self.batch_size = 1
        ctx = mx.gpu(9)

        #sm = get_text_cnn_symbol(sentence_size,vec_size,batch_size,dropout=0.0)
        sm = load_text_cnn_symbol(location_symbol,self.batch_size)

        input_shapes = {'data': (batch_size, 1, sentence_size, vec_size)}
        arg_shape, out_shape, aux_shape = sm.infer_shape(**input_shapes)
        arg_names = sm.list_arguments()

        p = mx.nd.load(location_model)
        arg_dict = {}
        for k,v in p.items():
            print k
            if k.startswith('arg:'):
                k = k[4:]
                arg_dict[k] = v
        arg_arrays = []
        for name, s in zip(arg_names, arg_shape):
            if name in ['softmax_label', 'data']:
                arg_arrays.append(mx.nd.zeros(s, ctx))
            else:
                arg_arrays.append(arg_dict[name])  


        cnn_exec = sm.bind(ctx=ctx,args=arg_arrays)

        data = cnn_exec.arg_dict['data']
        label = cnn_exec.arg_dict['softmax_label']

        self.scorer = CNNModel(cnn_exec=cnn_exec, symbol=sm, data=data, label=label, param_blocks=None)
    
        self.vec = get_w2v(location_vec)
        self.sentence_size = sentence_size

    def conv(self,sentence): # 分词之后的结果
        w2v = self.vec
        sentences = [sentence.split()]
        sentences_padded = pad_sentences(sentences,self.sentence_size)
        x_vec = []
        for sent in sentences_padded:
            vec = []
            for word in sent:
                if word in w2v:
                    vec.append(w2v[word])
                else:
                    vec.append(w2v['</s>'])
                #print word,vec[-1][0]
            x_vec.append(vec)
        x_vec = np.array(x_vec)

        return x_vec

    def get_score(self,sentence):
        x_vec = self.conv(sentence.lower())
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.scorer.data[:] = x_vec
        self.scorer.cnn_exec.forward(is_train=False)
        res = self.scorer.cnn_exec.outputs[0].asnumpy()

        res= [res[0][0],res[0][1]]
        res = res[1]
        res = float(res)
        return res

    def get_scores(self,sentence):
        x_vec = self.conv(sentence.lower())
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.scorer.data[:] = x_vec
        self.scorer.cnn_exec.forward(is_train=False)
        res = self.scorer.cnn_exec.outputs[0].asnumpy()

        res= [res[0][0],res[0][1]]
        res = res[1]
        res = float(res)
        return res


if __name__ == '__main__':

    print 'x'

    location_model = 'model/cnn-0030.params'
    location_vec = '/home/baotengfei/comment.vec.dat'
    
    sentence = u'多少 人 是 被 封面 骗 进来 的'
    vec_size = 80
    sentence_size = 25
    #scorer = TextCNNScorer(sentence_size,vec_size,location_vec,location_model)

    
    #res = scorer.get_score(sentence)
    #print res


