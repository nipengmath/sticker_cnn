#coding:utf-8

import os

from scorer import TextCNNScorer
from batch_scorer import BatchTextCNNScorer

import json

def get_x(m):
    return m['data']

def process_score(location_dat,location_res,\
                 location_vec,location_model,max_cnt=-1):
    
    print location_model

    #location_model = 'model_9/cnn-0162.params'

    with open(location_vec) as f:
        f.readline()
        line = f.readline()
        vs = line.strip().split()
        num = len(vs)-1

    vec_size = num
    sentence_size = 25

    scorer = TextCNNScorer(sentence_size,vec_size,location_vec,location_model)

    f_t = open(location_dat,mode='r')
    f_r = open(location_res,mode='w')

    cnt = 0
    for line in f_t:
        cnt += 1
        if cnt % 1000 == 0:
            print cnt
        if max_cnt > 0 and cnt > max_cnt:
            break
        m = json.loads(line)
        cut_title = get_x(m)

        res = scorer.get_score(cut_title)

        m['score'] = res

        line = json.dumps(m,ensure_ascii=False)
        f_r.write(line.encode('utf-8')+'\n')


    f_t.close()
    f_r.close()



def process_batch_score(location_dat,location_res,\
                 location_vec,location_model,location_symbol,location_size=None,max_cnt=-1):
    
    print location_model

    #location_model = 'model_9/cnn-0162.params'

    with open(location_vec) as f:
        f.readline()
        line = f.readline()
        vs = line.strip().split()
        num = len(vs)-1

    vec_size = num
    sentence_size = 25

    if location_size != None and os.path.exists(location_size):
        with open(location_size,mode='r') as f:
            line = f.read()
            s = json.loads(line)
            sentence_size = s['sentence_size']

    print 'sentence_size',sentence_size

    batch_size = 200

    scorer = BatchTextCNNScorer(sentence_size,vec_size,location_vec,location_model,location_symbol,batch_size)

    f_t = open(location_dat,mode='r')
    f_r = open(location_res,mode='w')

    cnt = 0
    ms = []
    for line in f_t:
        cnt += 1
        if cnt % 1000 == 0:
            print cnt
        if max_cnt > 0 and cnt > max_cnt:
            break
        m = json.loads(line)
        ms.append(m)
        if len(ms) == batch_size:
            sentences = []
            for i in range(batch_size):
                sentences.append(get_x(ms[i]))
            res = scorer.get_scores(sentences)
            for i in range(batch_size):
                ms[i]['score'] = float(res[i])
            for m in ms:
                line = json.dumps(m,ensure_ascii=False)
                f_r.write(line.encode('utf-8')+'\n')
            ms = []
        #cut_title = get_x(m)
        #res = scorer.get_score(cut_title)
        #m['score'] = res
        #line = json.dumps(m,ensure_ascii=False)
        #f_r.write(line.encode('utf-8')+'\n')

    sentences = []
    cc = len(ms)
    for i in range(cc):
        sentences.append(get_x(ms[i]))
    res = scorer.get_scores(sentences)
    for i in range(cc):
        ms[i]['score'] = float(res[i])
    for m in ms:
        line = json.dumps(m,ensure_ascii=False)
        f_r.write(line.encode('utf-8')+'\n')

    f_t.close()
    f_r.close()




