import os
from os import listdir, getcwd
from os.path import join
import random
wd = getcwd()

'''
##delete the wrong images and labels!!!!:
images_folder_path=os.path.join(wd,'images')
image_files_path=os.listdir(images_folder_path)
num=0

for file_obj in image_files_path:
    file_path = os.path.join(images_folder_path, file_obj)
    label_path=file_path.replace('images','labels').replace('.jpg','.txt')
    label_file=open(label_path)
    first_line=label_file.readline()
    first_line=first_line.split(' ')
    if (len(first_line)!=2):
      num=num+1
      print (len(first_line))
      print(label_path)
      print('remove',label_path)
      print('remove',file_path) 
      #os.remove(label_path)
      #os.remove(file_path)
print(num)      
print('Completed! \n')
'''

# divide
train_percent=0.85
val_percent=0.1
test_percent=0.05

images_folder_path=os.path.join(wd,'images')
image_files_path=os.listdir(images_folder_path)
pick_train_number=int(len(image_files_path)*train_percent)
pick_val_number=int(len(image_files_path)*val_percent)


train_files=random.sample(image_files_path,pick_train_number)
val_test_files=list(set(image_files_path)^set(train_files))

val_files=random.sample(val_test_files,pick_val_number)
test_files=list(set(val_test_files)^set(val_files))

print('number of the train_set:',len(train_files),'\n')
print('number of the val_set:',len(val_files),'\n')
print('number of the test_set:',len(test_files),'\n')


#For validation images path:
val_path = open('Val.txt', 'w')

for file_obj in val_files:
    file_path = os.path.join(images_folder_path, file_obj)
    label_path=file_path.replace('images','labels').replace('.jpg','.txt')
    label_file=open(label_path)
    #print(label_path)
    for line in label_file:
      line=line.split(' ')
      #print (line)    
      if ((line[0]=='0' and line[1]=='0')):
        #print('label=0, Hat , Cloth \n')
        label=0
      elif ((line[0]=='0' and line[1]=='1')):
        #print('label=1, Hat , No Cloth \n')
        label=1
      elif (line[0]=='0' and line[1]=='2'):
        #print('label=2, Hat , Uncertain Cloth \n')
        label=2
      elif (line[0]=='1' and line[1]=='0'):
        #print('label=3, No hat , Cloth\n')
        label=3
      elif (line[0]=='1' and line[1]=='1'):
        #print('label=4, No hat , No Cloth \n')
        label=4
      elif (line[0]=='1' and line[1]=='2'):
        #print('label=5, No hat , Uncertain Cloth \n')
        label=5     
      elif (line[0]=='2' and line[1]=='0'):
        #print('label=6, Uncertain hat , Cloth \n')
        label=6  
      elif (line[0]=='2' and line[1]=='1'):
        #print('label=7, Uncertain hat , No Cloth \n')
        label=7  
      elif (line[0]=='2' and line[1]=='2'):
        #print('label=8, Uncertain hat , Uncertain Cloth \n')
        label=8
      else:
        #print('Wrong label!\n')
        break      
    val_path.write(file_path + ' '+str(label)+'\n')
val_path.close()
print('Completed Val! \n')

#For train images path:
train_path = open('Train.txt', 'w')
for file_obj in train_files:
    file_path = os.path.join(images_folder_path, file_obj)
    label_path=file_path.replace('images','labels').replace('.jpg','.txt')
    label_file=open(label_path)
    #print(label_path)
    for line in label_file:
      line=line.split(' ')
      #print (line)
      if ((line[0]=='0' and line[1]=='0')):
        #print('label=0, Hat , Cloth \n')
        label=0
      elif ((line[0]=='0' and line[1]=='1')):
        #print('label=1, Hat , No Cloth \n') #no samples
        label=1
      elif (line[0]=='0' and line[1]=='2'):
        #print('label=2, Hat , Uncertain Cloth \n')
        label=2
      elif (line[0]=='1' and line[1]=='0'):
        #print('label=3, No hat , Cloth\n')
        label=3
      elif (line[0]=='1' and line[1]=='1'):
        #print('label=4, No hat , No Cloth \n')
        label=4
      elif (line[0]=='1' and line[1]=='2'):
        #print('label=5, No hat , Uncertain Cloth \n')
        label=5     
      elif (line[0]=='2' and line[1]=='0'):
        #print('label=6, Uncertain hat , Cloth \n')
        label=6  
      elif (line[0]=='2' and line[1]=='1'):
        #print('label=7, Uncertain hat , No Cloth \n')
        label=7  
      elif (line[0]=='2' and line[1]=='2'):
        #print('label=8, Uncertain hat , Uncertain Cloth \n')
        label=8
      else:
        print('Wrong label! \n')
        print(line)
        print(label_path)
        break       
    train_path.write(file_path + ' '+str(label)+'\n')
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
      line=line.split(' ')
      #print (line)
      if ((line[0]=='0' and line[1]=='0')):
        #print('label=0, Hat , Cloth \n')
        label=0
      elif ((line[0]=='0' and line[1]=='1')):
        #print('label=1, Hat , No Cloth \n') #no samples
        label=1
      elif (line[0]=='0' and line[1]=='2'):
        #print('label=2, Hat , Uncertain Cloth \n')
        label=2
      elif (line[0]=='1' and line[1]=='0'):
        #print('label=3, No hat , Cloth\n')
        label=3
      elif (line[0]=='1' and line[1]=='1'):
        #print('label=4, No hat , No Cloth \n')
        label=4
      elif (line[0]=='1' and line[1]=='2'):
        #print('label=5, No hat , Uncertain Cloth \n')
        label=5     
      elif (line[0]=='2' and line[1]=='0'):
        #print('label=6, Uncertain hat , Cloth \n')
        label=6  
      elif (line[0]=='2' and line[1]=='1'):
        #print('label=7, Uncertain hat , No Cloth \n')
        label=7  
      elif (line[0]=='2' and line[1]=='2'):
        #print('label=8, Uncertain hat , Uncertain Cloth \n')
        label=8
      else:
        print('Wrong label! \n')
        print(line)
        print(label_path)
        break       
    test_path.write(file_path + ' '+str(label)+'\n')
    label_file.close()
test_path.close()

print('Completed Test! \n')
