# coding:utf8
# 批量重命名文件
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
            filePath = Olddir
            Newname = Olddir.replace('-','_',1)
            print(filePath)
            print(Newname)
            os.rename(filePath, Newname)    #重命名

path = r"./+++/"
#path = r"./MM/"
reset(path)