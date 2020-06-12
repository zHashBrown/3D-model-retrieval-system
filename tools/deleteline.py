# coding:utf8
# 用这个文件来删除某文件的某一行
import os

def reset(path):
    i = 0
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件

        Olddir = os.path.join(path, files)  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            reset(Olddir)
        else:
            i = i + 1
            filename = os.path.splitext(files)[0]  # 文件名
            filetype = os.path.splitext(files)[1]  # 文件扩展名
            filePath = path + '/' + filename + filetype
            print(filePath)
            alter(filePath, "o ")


def alter(file, old_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        i=0
        for line in f1:
            i=i+1
            if i == 4 and old_str in line: # 第四行并且字符o在该行里
                print(line)
                continue
            else:
                f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


path = r"./ModelNet40/"
reset(path)