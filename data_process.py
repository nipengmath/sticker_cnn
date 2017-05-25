# -*- coding: utf-8 -*-

import codecs
import json
import random


def gen_corpus_ex():
    path = 'corpus.dat'
    f1 = codecs.open('location.train', 'w')
    f2 = codecs.open('location.test', 'w')


    with codecs.open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            r = random.random()
            ff = f1 if r > 0.2 else f2

            r = random.random()
            label = 1 if r >= 0.5 else 0

            row = {'x': line, 'z': label}
            ff.write('%s\n' %json.dumps(row))
    f1.close()
    f2.close()



if __name__ == '__main__':
    print 'ok'
    gen_corpus_ex()
