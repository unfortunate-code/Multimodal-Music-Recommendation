import os
from os import listdir
from os.path import isfile, join
import imghdr

def fix_image_files():
    dir_name = '../Data/images'
    files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    count = 0
    total_count = 0
    for file in files:
        if file.startswith('../Data/images/song'): continue
        if file.split('.')[-1] == 'png' or file.split('.')[-1] == 'jpg':
            continue
        if not imghdr.what(file):
            total_count += 1
            print(file)
    print(total_count)

if __name__ == '__main__':
    fix_image_files()
