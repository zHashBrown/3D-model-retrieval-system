from flask import Flask, session, render_template, request
from shutil import copyfile
import shutil
import pymysql
import bcrypt

import demo
import model
from retrieval import *


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.chdir(parentdir)  # 切换工作目录至外层

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tmp_path = './gui/static/tmp/'
ckptfile = 'model/model.ckpt-133000'
cloud_path = './gui/static/user_upload/'
convert_path = './gui/static/convert/'
lasttime = time.time()

if not os.path.exists(os.getcwd() + tmp_path):
    os.makedirs(os.getcwd() + tmp_path)
shutil.rmtree(tmp_path)  # 删除临时文件夹和文件夹下所有文件
os.mkdir(tmp_path)

if not os.path.exists(os.getcwd() + convert_path):
    os.makedirs(os.getcwd() + convert_path)
shutil.rmtree(convert_path)  # 删除转换文件夹和文件夹下所有文件
os.mkdir(convert_path)

'''预加载模型'''
with tf.Graph().as_default():
    view_ = tf.compat.v1.placeholder('float32', shape=(None, 12, 227, 227, 3), name='im0')
    y_ = tf.compat.v1.placeholder('int64', shape=None, name='y')
    keep_prob_ = tf.compat.v1.placeholder('float32')
    fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_, False)
    tf.compat.v1.get_variable_scope().reuse_variables()
    fc7 = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_, True)
    prediction = model.classify(fc8)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1000)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    saver.restore(sess, ckptfile)
    print('restore variables done')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zx164554'   # session密钥
app.config['UPLOAD_FOLDER'] = tmp_path  # 待检索文件上传到临时文件夹


@app.route('/', methods=['GET', 'POST', 'PUT'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'], strict_slashes=False)
def upload():
    file_dir = app.config['UPLOAD_FOLDER']  # 拼接成合法文件夹地址
    f = request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f:
        fname = f.filename
        print("上传文件：", fname)
        filename = os.path.basename(fname).replace(" ", "")  # 去掉空格！
        f.save(os.path.join(file_dir, filename))    # 保存文件到tmp目录
        demo.main(file_dir + filename)
        return filename
    return ''


@app.route('/retrieval', methods=['GET', 'POST', 'PUT'])  # 不使用/retrieval/，否则或导致访问静态资源自动添加相对路径 GET/retrieval/static/……
def retrieval_index():  # 出现问题要清空cookies，或者改访问静态资源的代码为绝对路径
    return render_template('retrieval.html')


@app.route('/retrieval_data/<filepath>/<load_ranges>', methods=['GET', 'POST', 'PUT'])
def retrieval_data(filepath, load_ranges):
    filename = filepath
    print('检索文件', filename)
    load_ranges = load_ranges.split(",")
    print('检索范围', load_ranges)
    ret = demo.retrival_data(filename, load_ranges,
                             view_, y_, keep_prob_, sess, prediction, fc7)
    return str(ret)[1:-1]


@app.route('/uptocloud/<filepath>', methods=['GET', 'POST', 'PUT'])
def uptocloud(filepath):
    global lasttime
    interval = time.time() - lasttime
    if interval < 10:
        ret = '操作过于频繁，请' + str(10 - interval).split('.')[0] + '秒后再试'
        return ret
    filename = os.path.basename(filepath)  # 带后缀
    name = filename.split('.')[0]  # 不带后缀
    target_path = cloud_path + 'tmp-' + str(int(time.time())) + '-' + name + '.obj'  # 带时间戳防止重名
    copyfile(tmp_path + name + '.obj', target_path)
    print("保存上传文件至云端：", target_path)
    lasttime = time.time()
    return '上传成功！模型经管理员审核后添加入库'


@app.route('/displayall/<load_range>', methods=['GET', 'POST', 'PUT'])
def displayall(load_range):
    print('用户查询：', load_range)
    ret = demo.getnames(load_range)
    return str(ret)[1:-1]


@app.route('/convert', methods=['GET', 'POST', 'PUT'])
def convert():
    return render_template('convert.html')


@app.route('/file_convert/<target_format>', methods=['GET', 'POST', 'PUT'])
def file_convert(target_format):
    file_dir = convert_path
    f = request.files['convert_file']  # 从表单的file字段获取文件，convert_file为该表单的name值
    if f:
        fname = f.filename
        print("用户转换：", fname)
        filename = os.path.basename(fname).replace(" ", "")  # 去掉空格！
        f.save(os.path.join(file_dir, filename))    # 保存文件到convert_file目录
        demo.convert(file_dir, filename, target_format)
        name = filename.rsplit('.', 1)[0]
        filename = name + "." + target_format
        return filename
    return ''


@app.route('/login/<username>/<password>', methods=['GET', 'POST'])
def login(username, password):
    user = username.strip()
    pwd = password.strip()

    # 链接，指定ip地址和端口，本机上测试时ip地址可以写localhost或者自己的ip地址或者127.0.0.1，然后你操作数据库的时候的用户名，密码，要指定你操作的是哪个数据库，指定库名，还要指定字符集。不然会出现乱码
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='', database='flask_login',
                           charset='utf8')
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 返回字典数据类型
    sql = "select * from user where username=%s;"
    res = cursor.execute(sql, [user])  # 防sql注入
    # 取到查询结果

    if res:
        print('用户存在')
        ret1 = cursor.fetchone()  # 取一条
        if bcrypt.checkpw(pwd.encode(), ret1['password'].encode()):
            print('登录成功')
            session['username'] = 'admin123'
            cursor.close()  # 关闭游标
            conn.close()  # 关闭连接
            return "success"
        else:
            cursor.close()  # 关闭游标
            conn.close()  # 关闭连接
            print('密码错误')
            return '密码错误'
    else:
        cursor.close()  # 关闭游标
        conn.close()  # 关闭连接
        print('用户不存在')
        return '用户不存在'


@app.route('/admin', methods=['GET', 'POST', 'PUT'])
def admin():
    if session.get('username'):
        return render_template('admin.html')
    else:
        return '未登录'


@app.route('/get_tmp_files', methods=['GET', 'POST', 'PUT'])
def get_tmp_files():
    ret = demo.get_tmp_files()
    return str(ret)[1:-1]


@app.route('/add_model/<add_model_name>/<add_model_class>', methods=['GET', 'POST', 'PUT'])
def add_model(add_model_name, add_model_class):
    print('添加模型', add_model_name, '到类别', add_model_class)
    demo.deal_feature_add(add_model_name, add_model_class,
                          view_, y_, keep_prob_, sess, prediction, fc7)
    return 'ok'


@app.route('/delete_model/<delete_model_name>', methods=['GET', 'POST', 'PUT'])
def delete_model(delete_model_name):
    print('删除模型', delete_model_name)
    demo.delete_model(delete_model_name)
    return 'ok'


@app.route('/clear_session')
def clear_session():
    print(session.get('username'), '管理员登出')
    session.clear()
    return 'success'


@app.route('/train', methods=['GET', 'POST', 'PUT'])
def train():
    return render_template('train.html')


@app.route('/start_train/<num_classes>')
def start_train(num_classes):
    demo.train_model(num_classes)
    return 'success'


if __name__ == '__main__':
    app.run(debug=False)
