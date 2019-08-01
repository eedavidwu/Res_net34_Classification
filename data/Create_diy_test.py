import os
from os import listdir, getcwd
from os.path import join
import random
wd = getcwd()



images_testing_path=os.path.join(wd,'test_img')
test_files=os.listdir(images_testing_path)


#For test images path:
test_path = open('Test.txt', 'w')
for file_obj in test_files:
    file_path = os.path.join(images_testing_path, file_obj)
    label_path=file_path.replace('test_img','test_label').replace('.jpg','.txt')
    label_file=open(label_path)
    #print(label_path)
    for line in label_file:
      line=line#.split(' ')
    test_path.write(file_path + ' '+str(line)+'\n')
    label_file.close()
test_path.close()
print('Completed Test! \n')


