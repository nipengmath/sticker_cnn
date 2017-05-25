#coding:utf-8

import json

def gen_tsv(location_dat,location_tsv,z,min_score=0.5):
    f_t = open(location_dat,mode='r')
    f_v = open(location_tsv,mode='w')
    
    for line in f_t:
        m = json.loads(line)
        title = m['text']
        s = m['score']
        i = m['id']
        g = 'http://admin.bytedance.com/crawl/article/article_detail/?id='+str(i)
        line = '%s\t%s\t%s'%(g,title,s)
        #if s < 0.5:
        #if s > 0.5:
        if z == 1 and s < min_score:
            continue
        if z == 0 and s > min_score:
            continue
        f_v.write(line.encode('utf-8')+'\n')

    f_t.close()
    f_v.close()





