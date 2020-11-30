import os
import sys

print(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
os.chdir('./cat')
l = os.listdir('.')
for i in range(len(l)):
    os.rename(l[i], 'cat_'+str(i))
    os.rename('../../대표이미지/'+l[i]+'.jpg','../../대표이미지/cat_'+str(i)+'.jpg' )
    print(l[i], i)