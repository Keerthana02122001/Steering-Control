import matplotlib.pyplot as plt

print('Setting up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from sklearn.model_selection import train_test_split

#### STEP 1
path ='myData'
data = importDataInfo(path)

#### STEP 2 - Visualization and distribution of data
balanceData(data,display=False)  ### if graph needed to be displayed, make the display as true or remover display label

##STEP 3
imagesPath,steerings =loadData(path,data)
#print(imagesPath[0],steering[0])


##STEP 4 - train and test (split data)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('Total training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

#### STEP 5 - Augmentation of data

### STEP 6 - preprocessing

### STEP 7

### STEP 8
model = createModel()
model.summary()

### STEP 9 -- training model
history = model.fit(batchGen(xTrain,yTrain,10,1),steps_per_epoch=20,epochs=2,
          validation_data=batchGen(xVal,yVal,10,0),validation_steps=20)

### STEP 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()