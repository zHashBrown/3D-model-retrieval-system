import numpy as np
import os, sys, inspect
import tensorflow as tf
import time
import os
import os.path as osp
import sklearn.metrics as metrics
import heapq
from input import Dataset
import globals as g_
import model
import pandas as pd
from sklearn.decomposition import PCA
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', osp.dirname(sys.argv[0]) + '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '',
                           """finetune with a pretrained model""")

np.set_printoptions(precision=3)

class_dict = {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf',
              5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain',
              12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box', 17: 'guitar',
              18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand',
              24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink',
              30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet',
              36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}


def test(dataset, ckptfile, listfile_labels, load_ranges='11111111111111111111111111111111111111111'):
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size
    data_size = dataset.size()

    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)

        view_ = tf.compat.v1.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.compat.v1.placeholder('int64', shape=None, name='y')
        keep_prob_ = tf.compat.v1.placeholder('float32')

        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_, False)
        tf.compat.v1.get_variable_scope().reuse_variables()
        fc7 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_, True)

        prediction = model.classify(fc8)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1000)

        init_op = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        saver.restore(sess, ckptfile)
        print('restore variables done')

        step = startstep

        filenames = []
        features = []
        predictions = []
        labels = []

        ap = 0
        print("Start testing")
        print("Size:", data_size)

        filenames, features = loadh5()                # 检索、测试map时使用
        for batch_x, batch_y in dataset.batches(1):  # 多线程读取时不可保存单个feature、不可计算map、不可检索

            step += 1
            print('第', step, '个文件')
            #filename = listfile_labels[0].rsplit('/')[-1].split('.')[0] + ".off"  # 单线程单文件时可用这句
            #print("加载文件：", filename)

            #filename = listfile_labels[step-1][0].rsplit('/')[-1].split('.')[0] + ".off"      # 大批量单线程时可用这句
            #print("加载文件：", filename)

            #print("加载文件：", os.path.basename(listfile_labels))             # 测试不带标签的样本可用这句
            #i = random.randint(0, 10)
            #batch_x = batch_x[:, i:i+1, :, :, :]
            if batch_x.shape[1] != 12:
                print("开始复制")
                batch_x = np.concatenate((batch_x, batch_x), axis=1)
                batch_1 = batch_x
                batch_x = np.concatenate((batch_x, batch_x), axis=1)
                batch_x = np.concatenate((batch_x, batch_1), axis=1)
                batch_x = np.concatenate((batch_x, batch_x), axis=1)
            print(batch_x.shape)
            feed_dict = {view_: batch_x,
                         y_: batch_y,
                         keep_prob_: 1.0}

            # 预测
            start_time = time.time()
            pred = sess.run(prediction, feed_dict=feed_dict)
            print('done classify one data, time=', time.time() - start_time)

            # 得到特征
            start_time = time.time()
            feature = (sess.run(fc7, feed_dict={
                                    view_: batch_x,
                                    keep_prob_: 1.0}))
            print('done get one feature, time=', time.time() - start_time)

            # retrieval_list = retrieval(feature, filenames, features)

            index = retrieval(feature, filenames, features)                            # 测试mAP
            ap += compute_ap(index, batch_y)                                      # 测试mAP


            #filenames.append(filename)
            #features.append(feature.tolist())
            predictions.extend(pred.tolist())  # 大批量预测   pred类型：<class 'numpy.ndarray'>
            labels.extend(batch_y.tolist())  # batch_y类型：<class 'numpy.ndarray'>

        map_score = ap / data_size                                      # 测试mAP
        print('mAP：', map_score)                                       # 测试mAP
        print('predictions:', predictions)
        acc = metrics.accuracy_score(labels, predictions)
        print('acc:', acc * 100)
        #return retrieval_list

        # 取出所有参与训练的参数
        # params = tf.compat.v1.trainable_variables()
        # print("Trainable variables:------------------------")           # 写报告的时候输出

        # 循环列出参数
        # for idx, v in enumerate(params):
        #    print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))





'''
PCA降维，需要样本数大于特征数，单个样本无法主成分分析

def P(X):
    pca = PCA(n_components=128)
    pca.fit(X)
    # PCA(copy=True, n_components=2, whiten=False)
    print(pca.explained_variance_ratio_)
'''

def compute_ap(rank_list, label):
    '''
    计算测试集中每个对象的ap值
    rank_list:查询图像返回的结果
    label:当前样本的类别
    pos_set:数据库中与查询图像相似的结果
    '''
    if label == 0:
        pos_set = range(0, 100)
    elif label == 1:
        pos_set = range(100, 150)
    elif label == 2:
        pos_set = range(150, 250)
    elif label == 3:
        pos_set = range(250, 270)
    elif label == 4:
        pos_set = range(270, 370)
    elif label == 5:
        pos_set = range(370, 470)
    elif label == 6:
        pos_set = range(470, 490)
    elif label == 7:
        pos_set = range(490, 590)
    elif label == 8:
        pos_set = range(590, 690)
    elif label == 9:
        pos_set = range(690, 710)
    elif label == 10:
        pos_set = range(710, 730)
    elif label == 11:
        pos_set = range(730, 750)
    elif label == 12:
        pos_set = range(750, 836)
    elif label == 13:
        pos_set = range(836, 856)
    elif label == 14:
        pos_set = range(856, 942)
    elif label == 15:
        pos_set = range(942, 962)
    elif label == 16:
        pos_set = range(962, 1062)


    elif label == 17:
        pos_set = range(1062, 1162)
    elif label == 18:
        pos_set = range(1162, 1182)
    elif label == 19:
        pos_set = range(1182, 1202)
    elif label == 20:
        pos_set = range(1202, 1222)
    elif label == 21:
        pos_set = range(1222, 1322)
    elif label == 22:
        pos_set = range(1322, 1422)
    elif label == 23:
        pos_set = range(1422, 1508)
    elif label == 24:
        pos_set = range(1508, 1528)
    elif label == 25:
        pos_set = range(1528, 1628)
    elif label == 26:
        pos_set = range(1628, 1728)
    elif label == 27:
        pos_set = range(1728, 1748)
    elif label == 28:
        pos_set = range(1748, 1848)
    elif label == 29:
        pos_set = range(1848, 1868)
    elif label == 30:
        pos_set = range(1868, 1968)
    elif label == 31:
        pos_set = range(1968, 1988)
    elif label == 32:
        pos_set = range(1988, 2008)
    elif label == 33:
        pos_set = range(2008, 2108)
    elif label == 34:
        pos_set = range(2108, 2128)
    elif label == 35:
        pos_set = range(2128, 2228)
    elif label == 36:
        pos_set = range(2228, 2328)
    elif label == 37:
        pos_set = range(2328, 2428)
    elif label == 38:
        pos_set = range(2428, 2448)
    elif label == 39:
        pos_set = range(2448, 2468)
    '''把小的模型挑出来用下面的
    elif label == 17:
        pos_set = range(1062, 1159)
    elif label == 18:
        pos_set = range(1159, 1179)
    elif label == 19:
        pos_set = range(1179, 1199)
    elif label == 20:
        pos_set = range(1199, 1219)
    elif label == 21:
        pos_set = range(1219, 1318)
    elif label == 22:
        pos_set = range(1318, 1418)
    elif label == 23:
        pos_set = range(1418, 1504)
    elif label == 24:
        pos_set = range(1504, 1524)
    elif label == 25:
        pos_set = range(1524, 1624)
    elif label == 26:
        pos_set = range(1624, 1724)
    elif label == 27:
        pos_set = range(1724, 1743)
    elif label == 28:
        pos_set = range(1743, 1843)
    elif label == 29:
        pos_set = range(1843, 1863)
    elif label == 30:
        pos_set = range(1863, 1963)
    elif label == 31:
        pos_set = range(1963, 1983)
    elif label == 32:
        pos_set = range(1983, 2003)
    elif label == 33:
        pos_set = range(2003, 2103)
    elif label == 34:
        pos_set = range(2103, 2123)
    elif label == 35:
        pos_set = range(2123, 2223)
    elif label == 36:
        pos_set = range(2223, 2322)
    elif label == 37:
        pos_set = range(2322, 2422)
    elif label == 38:
        pos_set = range(2422, 2442)
    elif label == 39:
        pos_set = range(2442, 2461)
    '''

    ap = 0.0
    intersect_size = 0.0
    for i in range(len(rank_list)):
        if rank_list[i] in pos_set:
            intersect_size += 1
            precision = intersect_size / (i + 1)    # 查准率
            ap += precision
    ap = ap / len(rank_list)
    print('ap：', ap)
    return ap


def read_lists(listfile_labels):
    if not listfile_labels:
        print("No Input!!!!!")
        exit(0)
    elif isinstance(listfile_labels[0], list):
        listfiles, labels = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    else:
        listfiles, labels = zip(*[(listfile_labels[0], int(listfile_labels[1]))])

    return listfiles, labels

'''
# 指定加载
def loadh5(load_ranges):
    start_time = time.time()
    filenames = []
    features = []
    #load_ranges = []
    for load_range in load_ranges:
        df = pd.read_hdf('./features/' + load_range + '.h5', key='filename')
        filenames.extend(df)
        df = pd.read_hdf('./features/' + load_range + '.h5', key='feature')
        features.extend(df)

    print('Load h5, time=', time.time() - start_time)
    print('共加载', len(filenames), '个模型')

    return filenames, features

# 字典遍历
def loadh5():
    filenames = []
    features = []
    start_time = time.time()
    for r in range(40):
        m = class_dict[r]
        df = pd.read_hdf('features/' + m + '.h5', key='filename')
        filenames.extend(df)
        df = pd.read_hdf('features/' + m + '.h5', key='feature')
        features.extend(df)
    print('Load h5, time=', time.time() - start_time)
    print('共加载', len(filenames), '个模型')
    return filenames, features
'''


# 加载test
def loadh5():
    filenames = []
    features = []
    start_time = time.time()
    df = pd.read_hdf('test57000.h5', key='filename')
    filenames.extend(df)
    df = pd.read_hdf('test57000.h5', key='feature')
    features.extend(df)
    print('Load h5, time=', time.time() - start_time)
    print('共加载', len(filenames), '个模型')
    return filenames, features


def feature_add(filename, feature, model_class):#   考虑把model_class直接用str表示,替换class_dict
    filenames = []
    features = []
    df = pd.read_hdf('./features/' + class_dict[model_class] + '.h5', key='filename')
    filenames.extend(df)
    df = pd.read_hdf('./features/' + class_dict[model_class] + '.h5', key='feature')
    features.extend(df)

    filenames.append(filename)
    print(len(feature[0]))
    features.append(feature[0])

    store = pd.HDFStore(class_dict[model_class]+'.h5', 'w')
    store.put('filename', pd.Series(filenames))
    store.put('feature', pd.Series(features))
    return filenames, features


def retrieval(model_feature, filenames, features):
    dist = []
    retrieval_list = []
    start_time = time.time()

    renum = 10
    for feature in features:  # 计算欧氏距离，可不用马氏，后续人工选择类别，全遍历可用马氏，但样本数要大于4069
        dist.append(np.linalg.norm(model_feature - feature))

    index = map(dist.index, heapq.nsmallest(renum, dist))  # 使用map执行list的index方法，返回该距离第一次出现的索引
    index = list(index)
    print("返回最相近的", renum, "个模型:", str(index))
    for i in index:
        retrieval_list.append(filenames[i])
    print(retrieval_list)
    print('done retrieval one data, time=', time.time() - start_time)
    # return retrieval_list
    return index                                                 #计算map返回index


def main(argv):  # 预测
    start_time = time.time()
    listfile_labels = np.loadtxt(g_.TEST_LOL, dtype=str).tolist()
    listfiles, labels = read_lists(listfile_labels)
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=g_.NUM_VIEWS)
    test(dataset, FLAGS.weights, listfile_labels)

    duration = time.time() - start_time
    print('ALL, time=', duration)


if __name__ == '__main__':
    main(sys.argv)

#     python map_test.py --weights=model/model.ckpt-57000
#     python map_test.py --weights=model/model.ckpt-107000
#     python map_test.py --weights=model/model.ckpt-133000
