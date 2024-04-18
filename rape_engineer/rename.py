import os


# 这种的只能处理 一个样本数据的 图片名字， 有待改进
def rename(path):
    filelist = os.listdir("D:/rape_flower_data")  # 获取指定的文件夹包含的文件或文件夹的名字的列表
    print(filelist)
    total_num = len(filelist)  # 获取文件夹内所有文件个数

    i = 0  # 图片名字从 0 开始
    c = 0
    for item in filelist:  # 遍历这个文件夹下的文件,即 图片
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(os.path.abspath(path), str(i) + '.jpg')

            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
                c = c + 1
            except:
                continue
    print('total %d to rename & converted %d jpgs' % (total_num, i))
    print('total %d to rename & converted %d jpgs' % (total_num, c))


if __name__ == '__main__':
    path = 'D:/rape_flower_data/'
    rename(path)

