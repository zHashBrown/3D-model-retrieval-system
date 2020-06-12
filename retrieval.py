import numpy as np
import os, sys, inspect
import tensorflow as tf
import time
import os
import heapq
from input import Dataset
import globals as g_
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('weights', '',
                           """finetune with a pretrained model""")

np.set_printoptions(precision=3)


def test(dataset, listfile_labels, load_ranges,
         view_, y_, keep_prob_, sess, prediction, fc7, flag_add=False, add_model_class='None'):
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size
    data_size = dataset.size()

    filenames = []
    features = []
    predictions = []
    labels = []

    print("Start testing")
    print("Size:", data_size)

    filenames, features = loadh5(load_ranges)  # 检索、测试map时使用
    for batch_x, batch_y in dataset.batches(1):  # 多线程读取时不可保存单个feature

        filename = os.path.basename(listfile_labels)
        print("加载文件：", filename)  # 测试不带标签的样本可用这句

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
        feature = sess.run(fc7, feed_dict={
            view_: batch_x,
            keep_prob_: 1.0})
        print('done get one feature, time=', time.time() - start_time)
        retrieval_list = retrieval(feature, filenames, features)

        if flag_add == True and add_model_class != 'None':
            feature_add(filename, feature, add_model_class)

        predictions.extend(pred.tolist())  # 大批量预测   pred类型：<class 'numpy.ndarray'>
        labels.extend(batch_y.tolist())  # batch_y类型：<class 'numpy.ndarray'>

    print('predictions:', predictions)
    return retrieval_list


def read_lists(listfile_labels):
    if not listfile_labels:
        print("No Input!!!!!")
        exit(0)
    elif isinstance(listfile_labels[0], list):
        listfiles, labels = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    else:
        listfiles, labels = zip(*[(listfile_labels[0], int(listfile_labels[1]))])

    return listfiles, labels


def loadh5(load_ranges):
    start_time = time.time()
    filenames = []
    features = []
    for load_range in load_ranges:
        df = pd.read_hdf('./features/' + load_range + '.h5', key='filename')
        filenames.extend(df)
        df = pd.read_hdf('./features/' + load_range + '.h5', key='feature')
        features.extend(df)

    print('Load h5, time=', time.time() - start_time)
    print('共加载', len(filenames), '个模型')

    return filenames, features


def feature_add(filename, feature, add_model_class):
    filenames = []
    features = []
    df = pd.read_hdf('./features/' + add_model_class + '.h5', key='filename')
    filenames.extend(df)
    df = pd.read_hdf('./features/' + add_model_class + '.h5', key='feature')
    features.extend(df)

    filenames.append(filename)
    features.append(feature[0])

    store = pd.HDFStore('./features/' + add_model_class + '.h5', 'w')
    store.put('filename', pd.Series(filenames))
    store.put('feature', pd.Series(features))
    store.close()


def retrieval(model_feature, filenames, features):
    dist = []
    retrieval_list = []
    start_time = time.time()

    for feature in features:  # 计算欧氏距离，可不用马氏，后续人工选择类别，全遍历可用马氏，但样本数要大于4069
        dist.append(np.linalg.norm(model_feature - feature))

    index = map(dist.index, heapq.nsmallest(45, dist))  # 使用map执行list的index方法，返回该距离第一次出现的索引
    index = list(index)
    print("返回最相近的45个模型:", str(index))
    for i in index:
        retrieval_list.append(filenames[i])
    print(retrieval_list)
    print('done retrieval one data, time=', time.time() - start_time)
    return retrieval_list


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
