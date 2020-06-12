import numpy as np
import os, sys, inspect
import tensorflow as tf
import time
import os
import sklearn.metrics as metrics
from input import Dataset
import globals as g_
import model
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '',
                           """finetune with a pretrained model""")

np.set_printoptions(precision=3)


def test():
    V = g_.NUM_VIEWS
    with tf.Graph().as_default():

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
        ckptfile = FLAGS.weights

        saver.restore(sess, ckptfile)
        print('restore variables done')
        for i in range(1):

            path = g_.TEST_LOL
            print(path)
            listfile_labels = np.loadtxt(path, dtype=str).tolist()
            listfiles, labels = read_lists(listfile_labels)
            dataset = Dataset(listfiles, labels, subtract_mean=False, V=g_.NUM_VIEWS)
            data_size = dataset.size()
            step = 0

            filenames = []
            features = []
            predictions = []
            labels = []

            for batch_x, batch_y in dataset.batches(1):  # 多线程读取时不可保存单个feature
                step += 1
                filename = listfile_labels[step-1][0].rsplit('/')[-1].split('.')[0] + ".off"      # 大批量单线程时可用这句
                print("加载模型：", filename)
                feed_dict = {view_: batch_x,
                             y_: batch_y,
                             keep_prob_: 1.0}
                # 预测
                pred = sess.run(prediction, feed_dict=feed_dict)

                # 得到特征
                feature = (sess.run(fc7, feed_dict={            # feature:[[0.    0.  ...]]，为(1,4069),不是(4069,)
                                        view_: batch_x,
                                        keep_prob_: 1.0}))

                filenames.append(filename)          # 单线程所以append
                features.append(feature[0].tolist())
                predictions.append(pred.tolist())   # 大批量预测 pred类型：<class 'numpy.ndarray'>
                labels.append(batch_y.tolist())     # batch_y类型：<class 'numpy.ndarray'>

            acc = metrics.accuracy_score(labels, predictions)
            print('acc:', acc * 100)
            save_h5(filenames, features)


def save_h5(filenames, features):
    x = pd.Series(filenames)
    x.to_hdf('test.h5', key='filename')
    y = pd.Series(features)
    y.to_hdf('test.h5', key='feature')


def read_lists(listfile_labels):
    if not listfile_labels:
        print("No Input!!!!!")
        exit(0)
    elif isinstance(listfile_labels[0], list):
        listfiles, labels = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    else:
        listfiles, labels = zip(*[(listfile_labels[0], int(listfile_labels[1]))])
    return listfiles, labels


def main(argv):
    start_time = time.time()
    test()
    print('ALL, time=', time.time() - start_time)


if __name__ == '__main__':
    main(sys.argv)

#    python gen_test_features.py --weights=model/model.ckpt-133000
#    python gen_test_features.py --weights=model/model.ckpt-57000
