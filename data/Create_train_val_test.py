import os
from os import listdir, getcwd
from os.path import join
import random
wd = getcwd()

# divide
train_percent=0.95
val_percent=0
test_percent=0.05

#images_testing_path=os.path.join(wd,'test_img')
images_folder_path=os.path.join(wd,'images')
image_files_path=os.listdir(images_folder_path)
pick_train_number=int(len(image_files_path)*train_percent)

train_files=random.sample(image_files_path,pick_train_number)
#val_files=list(set(image_files_path)^set(train_files))
#test_files=os.listdir(images_testing_path)
test_files=list(set(image_files_path)^set(train_files))

print('number of the train_set:',len(train_files),'\n')
#print('number of the val_set:',len(val_files),'\n')
print('number of the test_set:',len(test_files),'\n')
print('train+test:',len(train_files)+len(test_files))

'''
#For validation images path:
val_path = open('Val.txt', 'w')

for file_obj in val_files:
    file_path = os.path.join(images_folder_path, file_obj)
    label_path=file_path.replace('images','labels').replace('.jpg','.txt')
    label_file=open(label_path)
    #print(label_path)
    for line in label_file:
      line=line#.split(' ')
    val_path.write(file_path + ' '+str(line)+'\n')
val_path.close()
print('Completed Val! \n')
'''

#For train images path:
train_path = open('Train.txt', 'w')
for file_obj in train_files:
    file_path = os.path.join(images_folder_path, file_obj)
    label_path=file_path.replace('images','labels').replace('.jpg','.txt')
    label_file=open(label_path)
    #print(label_path)
    for line in label_file:
      line=line#.split(' ')
    train_path.write(file_path + ' '+str(line)+'\n')
    label_file.close()
train_path.close()
print('Completed Train! \n')


#For test images path:
test_path = open('Test.txt', 'w')
for file_obj in test_files:
    file_path = os.path.join(images_folder_path, file_obj)
    label_path=file_path.replace('images','labels').replace('.jpg','.txt')
    label_file=open(label_path)
    #print(label_path)
    for line in label_file:
      line=line#.split(' ')
    test_path.write(file_path + ' '+str(line)+'\n')
    label_file.close()
test_path.close()
print('Completed Test! \n')
