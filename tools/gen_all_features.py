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

class_dict = {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf',
              5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain',
              12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box', 17: 'guitar',
              18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand',
              24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink',
              30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet',
              36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}


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
        for i in range(40):
        #for i in range(1):

            path = './data/view/class_path/' + class_dict[i] + '.txt'
            #path = './data/view/class_path/' + 'xbox' + '.txt'
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

            print("Start save")
            print(class_dict[i], "Class size:", data_size)
            for batch_x, batch_y in dataset.batches(1):  # 多线程读取时不可保存单个feature

                step += 1
                filename = listfile_labels[step-1][0].rsplit('/')[-1].split('.')[0] + ".off"      # 大批量单线程时可用这句
                feed_dict = {view_: batch_x,
                             y_: batch_y,
                             keep_prob_: 1.0}
                # 预测
                pred = sess.run(prediction, feed_dict=feed_dict)

                # 得到特征
                feature = (sess.run(fc7, feed_dict={            # feature:[[0.    0.  ...]]，为(1,4069),不是(4069,)
                                        view_: batch_x,
                                        keep_prob_: 1.0}))

                print(filename)
                filenames.append(filename)          # 单线程所以append
                features.append(feature[0].tolist())
                predictions.append(pred.tolist())   # 大批量预测 pred类型：<class 'numpy.ndarray'>
                labels.append(batch_y.tolist())     # batch_y类型：<class 'numpy.ndarray'>

            class_now = class_dict[int(batch_y[0])]
            acc = metrics.accuracy_score(labels, predictions)
            print(class_now, 'acc:', acc * 100)
            save_h5(filenames, features, class_now)


def save_h5(filenames, features, model_class):
    store = pd.HDFStore('features/'+model_class+'.h5', 'w')
    store.put('filename', pd.Series(filenames))
    store.put('feature', pd.Series(features))
    store.close()


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

#    python gen_all_features.py --weights=model/model.ckpt-133000
