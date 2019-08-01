# Resnet_Classification

## Create your dataset

 Put the image into the ./data/images/

 Put the label (txt file with same name) into the ./data/labels/



## Create the Train.txt, Val.txt, Test.txt

If you want to create the train.txt, Val.txt, Test.txt in random:

python Create_train_val_test.py

If you want to create the test in some determined folder:

python Create_diy_test.py



## Train your dataset

#### Firstly, choose the network your want to train in classifier_train.py:

model = resnet34(pretrained=False, modelpath=model_path)  

#### Secondly, change the  transform (size and operation in need) and the Avg_pooling:

input size : f x f, average pooling size: f/32

T.Resize((96,96)),

self.avgpool = nn.AvgPool2d(3, stride=1) 

#### Thirdly, change the  FC (2 class):

Fully connect layer in 'self.fc_hat = nn.Linear(512 * block.expansion, 2)'

#### Fourthly,  add the dropout or not depend on case: 

self.dropout=nn.Dropout(p=0.8)

#### Start train:

python classifier_train.py


