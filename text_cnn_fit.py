#coding:utf-8

from text_cnn import *

from text_iterator import TextIterator


def fit(m, text_iter,test_iter, batch_size,\
    optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200, root='model/'):

    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate
    updater = mx.optimizer.get_updater(opt)

    location_log = root+'model.log'
    print location_log
    f_log = open(location_log,mode='w')

    for iteration in range(epoch):
        ## 每一轮
        tic = time.time()
        num_correct = 0
        num_total = 0
        batch_num = text_iter.cnt/batch_size+1
        time_for_data = 0
        real_train_time = 0.   # 包括取数据的时间， 但是不包括算accuracy的时间
        for _ in range(batch_num):
            ## 每个batch
            train_tic = time.time()
            try:
                tic_ = time.time()
                batchX, batchY = text_iter.next_batch(batch_size)
                time_for_data += time.time() - tic_
            except Exception, e:
                print >> logs, "loading data error"
                print >> logs, repr(e)
                continue

            if batchX.shape[0] != batch_size:
                continue
            m.data1[:] = batchX
            m.data2[:] = batchX
            m.label[:] = batchY
            m.cnn_exec.forward(is_train=True)
            m.cnn_exec.backward()
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= max_grad_norm / norm
                updater(idx, grad, weight)
                grad[:] = 0.0
            real_train_time += time.time() - train_tic
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print >> logs, 'reset learning rate to %g' % opt.lr

        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)
        if (iteration + 1) % 1 == 0:
            prefix = root + 'cnn'
            m.symbol.save('%s-symbol.json' % prefix)
            save_dict = {'arg:%s' % k:v for k, v in m.cnn_exec.arg_dict.items() if k != 'embedding_weight'}
            save_dict.update({'aux:%s' % k:v for k, v in m.cnn_exec.aux_dict.items()if k != 'embedding_weight'} )
            param_name = '%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            print >> logs, 'Saved checkpoint to %s' % param_name

        if (iteration + 1) == epoch:
            save_dict_cpu = {k:v.copyto(mx.cpu()) for k,v in save_dict.items() if k != 'embedding_weight'}
            mx.nd.save(param_name+'.cpu',save_dict_cpu)
        num_correct = 0
        num_total = 0
        ps = []
        batch_num = test_iter.cnt/batch_size+1
        y_dev_batch = []
        for _ in range(batch_num):
            try:
                batchX, batchY = test_iter.next_batch(batch_size)
            except Exception,e:
                print >> logs, repr(e)
                continue
            if batchX.shape[0] != batch_size:
                continue
            y_dev_batch.extend(batchY)
            m.data1[:] = batchX
            m.data2[:] = batchX
            m.cnn_exec.forward(is_train=False)
            ps.extend(np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        cat_res = evaluate(y_dev_batch, ps)
        dev_acc = num_correct * 100 / float(num_total)

        print >> logs, 'Iter [%d] Train: Time: %.3fs, Real Train Time: %.3f, Data Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, real_train_time, time_for_data, train_acc)
        print >> logs, '--- Dev Accuracy thus far: %.3f' % dev_acc

        line = 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f' % (iteration, train_time, train_acc)

        f_log.write(line.encode('utf-8')+'\n')
        f_log.flush()

        line = '--- Dev Accuracy thus far: %.3f' % dev_acc
        f_log.write(line.encode('utf-8')+'\n')
        f_log.flush()

        for cat, res in cat_res.items():
            print >> logs, '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])

            line = '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
            f_log.write(line.encode('utf-8')+'\n')
            f_log.flush()
    f_log.close()
