#coding:utf-8

import json
import random

def conv(r):
    m = {}
    m['i'] = r['id']
    m['text'] = r['text']
    m['x'] = r['data']
    m['z'] = int(r['label'])
    return m

def merge_o(location0,r0, # Positive Instnace
          location1,r1, # Negative Instance
          location_res):
    ms = []
    with open(location0,mode='r') as f:
        for line in f:
            r = json.loads(line)
            rr = random.random()
            if rr > r0 and r['label'] == 0:
                continue
            z = r['label']
            z = int(z)
            if z == 1:
                print r['id'],r['text'],z
            m = conv(r,z)
            ms.append(m)
    with open(location1,mode='r') as f:
        for line in f:
            r = json.loads(line)
            rr = random.random()
            if rr > r1 and r['label'] == 1:
                continue
            z = r['label']
            z = int(z)
            if z == 0:
                print r['id'],r['text'],z
            m = conv(r,z)
            ms.append(m)
    random.shuffle(ms)
    with open(location_res,mode='w') as f:
        for m in ms:
            line = json.dumps(m,ensure_ascii=False)
            f.write(line.encode('utf-8')+'\n')

def sample(z,r,location_dat,location_res):
    f_t = open(location_dat,mode='r')
    f_r = open(location_res,mode='w')

    for line in f_t:
        m = json.loads(line)
        rr = random.random()
        if rr > r and int(m['label']) == z:
            continue
        f_r.write(line)

    f_t.close()
    f_r.close()

def merge(locations,location_res):
    ms = []
    for location in locations:
        with open(location,mode='r') as f:
            for line in f:
                r = json.loads(line)
                m = conv(r)
                ms.append(m)

    random.shuffle(ms)
    with open(location_res,mode='w') as f:
        for m in ms:
            line = json.dumps(m,ensure_ascii=False)
            f.write(line.encode('utf-8')+'\n')
            #f.write(line)



def count(location):
    m_c = {}
    with open(location) as f:
        for line in f:
            r = json.loads(line)
            z = r['z']
            c = m_c.get(z,0)
            m_c[z] = c+1
    print m_c


