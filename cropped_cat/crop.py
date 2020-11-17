import os

for i in range(1,92):
    comm = "python catfd.py -f cat/insta_{} -o outputcat/insta_{} -j -c".format(i,i)
    os.system(comm)