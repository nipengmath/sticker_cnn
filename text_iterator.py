#coding:utf-8

import json
import numpy as np

def get_vocab(location):
    id2v = []
    w2id = {}
    idx = 0
    num = 0
    with open(location) as f:
        for line in f:
            line = line.decode('utf8')
            xs = line.strip().split()
            if len(xs) == 2:
                continue
            if num == 0:
                num = len(xs)
            if num !=0 and len(xs) != num:
                #print line
                continue
            w2id[xs[0]] = idx
            idx += 1
            id2v.append(map(float,xs[1:]))
    return np.array(id2v), w2id



class TextIterator(object):

    def __init__(self, w2id, location, sentence_size):
        self.cnt = 0
        self.data_buffer = []
        self.label_buffer = []
        self.sentence_size = sentence_size
        with open(location) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                self.data_buffer.append([w2id[w] for w in data['x'].split() if w in w2id][:sentence_size])
                self.label_buffer.append(data['z'])
                self.cnt += 1
        self.index = 0
        self.pad_id = w2id['</s>']

    def pad(self, sentences):
        for i in xrange(len(sentences)):
            len_ = len(sentences[i])
            sentences[i] += [self.pad_id] * (self.sentence_size - len_)

    def next_batch(self, batch_size):
        data_batch = []
        label_batch = []
        for _ in xrange(batch_size):
            data_batch.append(self.data_buffer[self.index])
            label_batch.append(self.label_buffer[self.index])
            self.index = (self.index + 1) % self.cnt
        self.pad(data_batch)
        return np.array(data_batch), np.array(label_batch)


if __name__ == '__main__':

    print 'x'


    m_vec = get_w2v('vec.s')
    print 'm_vec',len(m_vec)

    it = TextIter(m_vec,'bait_title.ins',20)

    print it.cnt
    xs,ys = it.next_batch(10)
    print xs
    print ys

    it.close()
