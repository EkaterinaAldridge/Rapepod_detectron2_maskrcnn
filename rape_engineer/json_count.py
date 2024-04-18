import os
import json

if __name__ == '__main__':
    filePath = 'rape_json_count'
    rape_count = 0
    file_count = 0
    for root, dirs, files in os.walk(filePath):
        for filename in files:
            with open(filePath + '/' + filename) as json_data:
                result = json.load(json_data)
                one_json_list = result.get('shapes')
                temp = len(one_json_list)
                print('当前文件' + filename + '的果荚数量是：', temp)
                rape_count = rape_count + temp
                file_count = file_count + 1
    print('总的文件数量是：', file_count)
    print('总的果荚数量是：', rape_count)
