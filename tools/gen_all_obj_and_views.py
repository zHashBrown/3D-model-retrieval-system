import os

# 会在tmp文件夹生成txt文件，但是分类为-1
def main(file_path):

    filename = os.path.basename(file_path)  # 带后缀
    print(file_path)
    print(tmp_path + filename)
    # 生成views
    os.system(r'".\blender-2.79\blender.exe" '  # 这里注意反斜杠，应该是绝对路径或者 .\ 和 \          Windows特性^^
                  r'BlenderPhong/phong.blend '
                  r'--background '
                  r'--python BlenderPhong/phong.py '
                  r'-- ' + file_path                        # 参数不能有空格
                  + '  ' + tmp_path)
    
    # 将模型转换为obj形式
    os.system(r'".\blender-2.79\blender.exe" '
                  r'BlenderPhong/phong.blend '
                  r'--background '
                  r'--python BlenderPhong/convert.py '
                  r'-- ' + file_path
                  + '  ' + tmp_path)
    ''''''


def gen_offpath(path):
    i = 0
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件

        Olddir = os.path.join(path, files)  # 原来的文件路径
        if os.path.isdir(Olddir):# 如果是文件夹则跳过
            gen_offpath(Olddir)
        else:
            i = i + 1
            filename = os.path.splitext(files)[0]  # 文件名
            filetype = os.path.splitext(files)[1]  # 文件扩展名
            filepath = path+'\\'+filename + filetype
            print(filepath)
            f = open("off_path.txt", "a")  # 设置文件对象
            f.writelines(filepath+'\n')  # 直接将文件中按行读到list里，效果与方法2一样
            f.close()  # 关闭文件


if __name__ == '__main__':
    tmp_path = './gen/'
    if not os.path.exists(os.getcwd() + tmp_path):
        os.makedirs(os.getcwd() + tmp_path)
    datasetpath = r'E:\Users\lenovo\Desktop\Graduation_project\ModelNet40'
    gen_offpath(datasetpath)

    paths = []
    for line in open("off_path.txt", "r"):  # 设置文件对象并读取每一行文件
        paths.append(line[:-1])         # 将每一行文件加入到list中，去掉换行符
    for path in paths:
        main(path)

