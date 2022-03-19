from numpy import number
import pandas as pd

# TOTAL = 112550
TOTAL = 136098
SIZE = 10

# generate bash for all
# for i in range(0,TOTAL,SIZE):
#     print("python spotify_extract_ids.py {} {}".format(SIZE,i))



# find missing ids
PATH = "../Data/audio/song_map.txt"
df = pd.read_csv(PATH,sep='\t')
# print(len(df),'entries found')

list = df['id'].tolist()
list = [int(x) for x in list]
list.sort()

for index, id in enumerate(list):
    if id < TOTAL and list[index+1] - id > 1:
        begin = id
        end = list[index+1]
        size = end - id - 1
        if size > SIZE:
            # print("size > 10")
            for i in range(0, size, SIZE):
                subsize = SIZE if begin + SIZE < end else (end - begin)
                print("python spotify_extract_ids.py {} {}".format(subsize, begin))
                begin = begin + subsize
        else:       
            print("python spotify_extract_ids.py {} {}".format(size, begin))

