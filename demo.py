import shutil
from retrieval import *
from shutil import copyfile

tf.app.flags.DEFINE_string('path', '',
                           """待预测的模型或图片""")


def main(file_path):  # 从模型到检索

    tmp_path = './tmp/'
    gui_tmp_path = './gui/static/tmp/'
    if not os.path.exists(os.getcwd() + tmp_path):
        os.makedirs(os.getcwd() + tmp_path)
    shutil.rmtree(tmp_path)  # 删除该文件夹和文件夹下所有文件
    os.mkdir(tmp_path)
    # file_path = FLAGS.path  # 文件路径
    filename = os.path.basename(file_path).replace(" ", "")  # 带后缀，去掉空格
    copyfile(file_path, tmp_path + filename)
    ext = filename.split('.')[-1].lower()
    name = filename.rsplit('.', 1)[0]

    if ext == 'jpg' or ext == 'jpeg' or ext == 'png' or ext == 'bmp':
        with open('./tmp/' + name + '.' + ext + '.txt', 'w') as file_handle:  # 新建保存图像路径的txt
            file_handle.write('-1\n12\n')  # 写入
        for i in range(12):
            copyfile(file_path, tmp_path + name + '.' + str(i) + '.png')
            with open('./tmp/' + name + '.' + ext + '.txt', 'a') as file_handle:
                file_handle.write(tmp_path + name + '.' + str(i) + '.png' + '\n')  # 写入
    else:
        os.system(r'".\blender-2.79\blender.exe" '  # 这里注意反斜杠，应该是绝对路径或者 .\ 和 \          Windows特性^^
                  r'BlenderPhong/phong.blend '
                  r'--background '
                  r'--python BlenderPhong/phong.py '
                  r'-- ' + tmp_path + filename
                  + '  ' + tmp_path)
        os.system(r'".\blender-2.79\blender.exe" '
                  r'BlenderPhong/phong.blend '
                  r'--background '
                  r'--python BlenderPhong/convert.py '  # 将模型转换为obj形式
                  r'-- ' + tmp_path + filename
                  + '  ' + tmp_path)
        copyfile(tmp_path + name + '.obj', gui_tmp_path + name + '.obj')  # obj文件传到服务器文件夹下以显示


def retrival_data(filename, load_ranges,
                  view_, y_, keep_prob_, sess, prediction, fc7, flag_add=False, add_model_class='None'):
    tmp_path = './tmp/'
    txt_path = tmp_path + filename + '.txt'
    listfiles, labels = read_lists([txt_path, -1])  # 检测时不带label
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=g_.NUM_VIEWS)
    ret = test(dataset, filename, load_ranges,
               view_, y_, keep_prob_, sess, prediction, fc7, flag_add, add_model_class)
    return ret


def getnames(load_range):  # 获得该类别所有模型名称，仅显示在特征库里的而不是遍历文件夹
    filenames = []
    df = pd.read_hdf('features/' + load_range + '.h5', key='filename')
    filenames.extend(df)
    return filenames


# off 全OK
# ply 全OK
# stl 全OK
# obj .转ply有问题.转off有问题
# 3ds .转ply有问题.转off有问题
# fbx .转ply有问题.转off有问题
# x3d .转ply有问题.转off有问题
# 有问题的先转成stl再转成其他的
def convert(file_dir, filename, target_format):
    ext = filename.split('.')[-1].lower()
    if target_format == 'ply' or target_format == 'off':
        if ext == 'obj' or ext == '3ds' or ext == 'fbx' or ext == 'x3d':
            os.system(r'".\blender-2.79\blender.exe" '
                      r'BlenderPhong/phong.blend '
                      r'--background '
                      r'--python BlenderPhong/convert_all.py '  # 将模型转换为stl形式
                      r'-- ' + file_dir + filename
                      + '  ' + file_dir
                      + '  ' + 'stl')
            name = filename.rsplit('.', 1)[0]
            filename = name + '.stl'

    os.system(r'".\blender-2.79\blender.exe" '
              r'BlenderPhong/phong.blend '
              r'--background '
              r'--python BlenderPhong/convert_all.py '  # 将模型转换为stl形式
              r'-- ' + file_dir + filename
              + '  ' + file_dir
              + '  ' + target_format)
    return '转换完成'


def get_tmp_files():
    filenames = []
    user_upload_path = './gui/static/user_upload/'
    filelist = os.listdir(user_upload_path)  # 该文件夹下所有的文件（包括文件夹）
    for file in filelist:  # 遍历所有文件
        filenames.append(file)
    return filenames


def deal_feature_add(add_model_name, add_model_class,
                     view_, y_, keep_prob_, sess, prediction, fc7):
    user_upload_path = './gui/static/user_upload/'
    dstdir = './gui/static/models/obj/ModelNet40/' + add_model_class  # 目标文件夹

    beforepath = user_upload_path + add_model_name  # 改名前user_upload路径
    new_add_model_name = add_model_name.replace('tmp', add_model_class, 1)  # 新文件名
    afterpath = user_upload_path + new_add_model_name  # 改名后user_upload路径
    os.rename(beforepath, afterpath)

    main(afterpath)
    retrival_data(new_add_model_name, [add_model_class],  # 以addclass加载，并无意义
                  view_, y_, keep_prob_, sess, prediction, fc7, flag_add=True, add_model_class=add_model_class)

    # 移动文件
    if not os.path.isfile(afterpath):
        print("%s not exist!" % afterpath)
    else:
        fpath, fname = os.path.split(dstdir)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(afterpath, dstdir)  # 移动文件
        print("move %s -> %s" % (afterpath, dstdir))


def delete_model(delete_model_name):
    user_upload_path = './gui/static/user_upload/'
    m40path = './gui/static/models/obj/ModelNet40/'
    delete_model_class = delete_model_name.split('-')[0]

    if delete_model_class == 'tmp':
        delete_model_path = user_upload_path + delete_model_name
    else:
        delete_model_path = m40path + delete_model_class + '/' + delete_model_name
        delete_from_h5(delete_model_class, delete_model_name)

    if os.path.exists(delete_model_path):  # 如果文件存在
        os.remove(delete_model_path)
    else:
        print('no such file:%s' % delete_model_path)  # 则返回文件不存在


def delete_from_h5(delete_model_class, delete_model_name):
    filenames = []
    features = []
    df = pd.read_hdf('./features/' + delete_model_class + '.h5', key='filename')
    filenames.extend(df)
    df = pd.read_hdf('./features/' + delete_model_class + '.h5', key='feature')
    features.extend(df)

    index = filenames.index(delete_model_name)
    filenames.pop(index)
    features.pop(index)

    store = pd.HDFStore('./features/' + delete_model_class + '.h5', 'w')
    store.put('filename', pd.Series(filenames))
    store.put('feature', pd.Series(features))
    store.close()


def train_model(num_classes):
    import train_gui
    train_gui.main(sys.argv, num_classes)


if __name__ == '__main__':
    main('./None.None')

# 手动输入模型路径不可带空格，GUI自选时无恙，但要做去空格处理

# python demo.py --weights=model/model.ckpt-57000 --path=./range_hood_0015.off
# python demo.py --weights=model/model.ckpt-57000 --path=./airplane_0627.png
