import cv2
import random
import numpy as np
import time
import queue
import threading
import globals as g_
from concurrent.futures import ThreadPoolExecutor

W = H = 256

'''
读取图像，解决imread不能读取中文路径的问题
cv2.IMREAD_COLOR : 默认使用该种标识。加载一张彩色图片，忽视它的透明度。
cv2.IMREAD_GRAYSCALE : 加载一张灰度图。
cv2.IMREAD_UNCHANGED : 加载图像，包括它的Alpha通道
'''
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


class Shape:
    def __init__(self, list_file):
        with open(list_file, encoding='gbk') as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]

            #view_files = [l.strip().replace('/','.off/').replace('.off/','/',6) for l in f.readlines()]  # 读取原classes用这句

        self.views = self._load_views(view_files, self.V)
        self.done_mean = False


    def _load_views(self, view_files, V):
        views = []
        for f in view_files:
            im = cv_imread(f)
            im = cv2.resize(im, (W, H))
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
            assert im.shape == (W, H, 3), 'BGR!'
            im = im.astype('float32')
            views.append(im)
        views = np.asarray(views)  # (12, 256, 256, 3)
        return views

    def subtract_mean(self):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  # 没用这函数
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.views[:, :, :, i] -= mean_bgr[i]

            self.done_mean = True

    def crop_center(self, size=(227, 227)):  # 256->227???????????????????????????????????????
        w, h = self.views.shape[1], self.views.shape[2]
        wn, hn = size
        left = w // 2 - wn // 2
        top = h // 2 - hn // 2
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, left:right, top:bottom, :]  # (12, 227, 227, 3)


class Dataset:
    def __init__(self, listfiles, labels, subtract_mean, V):
        self.listfiles = listfiles
        self.labels = labels
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.V = V
        print('dataset inited')
        print('total size:', len(listfiles))

    def shuffle(self):
        z = list(zip(self.listfiles, self.labels))
        random.shuffle(z)
        self.listfiles, self.labels = [list(l) for l in zip(*z)]
        self.shuffled = True

    def batches(self, batch_size):
        for x, y in self._batches_fast(self.listfiles, batch_size):
            yield x, y

    def sample_batches(self, batch_size, n):  # 验证时用
        listfiles = random.sample(self.listfiles, n)
        for x, y in self._batches_fast(listfiles, batch_size):
            yield x, y

    def _batches(self, listfiles, batch_size):
        n = len(listfiles)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  # 没用这函数，用的多线程
        for i in range(0, n, batch_size):
            starttime = time.time()

            lists = listfiles[i: i + batch_size]
            x = np.zeros((batch_size, self.V, 227, 227, 3))
            y = np.zeros(batch_size)

            for j, l in enumerate(lists):
                s = Shape(l)
                s.crop_center()
                if self.subtract_mean:
                    s.subtract_mean()
                x[j, ...] = s.views
                y[j] = s.label

            print('load batch time:', time.time() - starttime, 'sec')
            yield x, y

    def _load_shape(self, listfile):
        s = Shape(listfile)
        s.crop_center()
        if self.subtract_mean:
            s.subtract_mean()
        return s

    def _batches_fast(self, listfiles, batch_size):  # train.txt 16
        subtract_mean = self.subtract_mean
        n = len(listfiles)

        def load(listfiles, q, batch_size):
            n = len(listfiles)
            with ThreadPoolExecutor(max_workers=16) as pool:  # 16线程！！！
                for i in range(0, n, batch_size):
                    sub = listfiles[i: i + batch_size] if i < n - 1 else [listfiles[-1]]
                    shapes = list(pool.map(self._load_shape, sub))
                    views = np.array([s.views for s in shapes])  # shapes的属性 views:(batch_size, 12, 227, 227, 3)
                    labels = np.array([s.label for s in shapes])  # shapes的属性
                    q.put((views, labels))

            # indicate that I'm done
            q.put(None)

        # This must be larger than twice the batch_size
        q = queue.Queue(maxsize=g_.INPUT_QUEUE_SIZE)  # 4*16

        # background loading Shapes process
        p = threading.Thread(target=load, args=(listfiles, q, batch_size))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()

        x = np.zeros((batch_size, self.V, 227, 227, 3))
        y = np.zeros(batch_size)

        for i in range(0, n, batch_size):
            item = q.get()
            if item is None:
                break
            x, y = item
            yield x, y

    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.listfiles)

# crop_center过程
