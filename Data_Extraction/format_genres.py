import json
 
filename = "../Data/genres.json"
f = open(filename)

data = json.load(f)

with open("../Data/audio/songs.txt",'w',encoding = 'utf-8') as f_out:
    for index, key in enumerate(data):
        key = key.split(" by ")
        line = key[0]+"\t"+key[1]+"\t"+str(index+1)+"\n"
        f_out.write(line)

f.close()