# praktikumML_314_321
Tugas Kelompok Praktikum Machine Learning

----_**Overview semua hasil dari modul keseluruhan**_------

Overview jurnal :
-Data yang digunakan pada jurnal yang berjudul Unveiling COVID-19 from CHEST X-Ray with
Deep Learning: A Hurdles Race with Small Data, yaitu data Covid-19 rontgen dada di salah satu rumah sakit darurat utama tepatnya di daerah italia utara
-preprocessing data yang dilakukan dataset pada jurnal ini memiliki tujuan untuk menghilangkan bias dalam data. Yang pertama menggunakan ekualisasi histogram guna menjamin dinamika gambar menjadi seragam di dalam data.kemudian melakukan segmentasi pada paru-paru guna mengelompokkan gambar dengan gambar yang ditampilkan hanya paru-paru saja, kemudian diburamkan pada tepian dengan menggunakan radisu 3 pixel guna mengatasi tepi yang tajam.
-Model yang digunakan pada jurnal rujukan yaitu ResNet-18, ResNet-50, COVID-Net, DenseNet-121

Overview dataset :
-Link Sumber dataset :https://www.kaggle.com/jtiptj/chest-xray-pneumoniacovid19tuberculosis
-Jumlah data gambar keseluruhan 7135 image, dan dibagi menjadi 3 yaitu train, test, val. jumlah nilai di dalamnya yaitu pada label train 6326 images, label test 771 images, dan label val 38 images.




# ---**Modul 2**---- 
- Jumlah splititng data set yaitu 
Data COVID19 Train : 460
Data COVID19 Validation : 10
Data COVID19 Test : 106
Data NORMAL Train : 1341
Data NORMAL Validation : 8
Data NORMAL Test : 234
Data PNEUMONIA Train : 3875
Data PNEUMONIA Validation : 8
Data PNEUMONIA Test : 390
Data TURBERCULOSIS Train : 650
Data TURBERCULOSIS Validation : 12
Data TURBERCULOSIS Test : 41
presentase spilt data train 80%
Presentaase split data test 19%
Presentase split data val 1%

Gather data :
import cv2
import numpy as np
#gather data train
train_data = []
train_label = []
for r, d, f in os.walk(train) :
    for file in f:
        if ".jpg" in file:
            imagePath = os.path.join(r, file)
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (150,150))
            train_data.append(image)
            label = imagePath.split(os.path.sep)[-2]
            train_label.append(label)
train_data = np.array(train_data)
train_label = np.array(train_label)

#Gather data validation
val_data = []
val_label = []
for r, d, f in os.walk(validation) :
    for file in f:
        if ".jpg" in file:
            imagePath = os.path.join(r, file)
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (150,150))
            val_data.append(image)
            label = imagePath.split(os.path.sep)[-2]
            val_label.append(label)
val_data = np.array(val_data)
val_label = np.array(val_label)

-Model yang digunakan sequential dengan menggunakan beberapa layers yaitu InputLayer, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
**-Model 1**
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d_6 (Conv2D)            (None, 150, 150, 16)      448       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 75, 75, 16)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 75, 75, 32)        4640      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 38, 38, 32)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 38, 38, 64)        18496     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 19, 19, 64)        0         
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 64)                0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               8320      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 129       
=================================================================
Total params: 32,033
Trainable params: 32,033
Non-trainable params: 0
_________________________________________________________________
None

**Matiriks Evaluasi model-1**
model 1
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10

**Model 2**
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d_9 (Conv2D)            (None, 150, 150, 16)      448       
_________________________________________________________________
average_pooling2d_3 (Average (None, 75, 75, 16)        0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 75, 75, 32)        4640      
_________________________________________________________________
average_pooling2d_4 (Average (None, 38, 38, 32)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 38, 38, 64)        18496     
_________________________________________________________________
average_pooling2d_5 (Average (None, 19, 19, 64)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 64)                0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_6 (Dense)              (None, 128)               8320      
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 129       
=================================================================
Total params: 32,033
Trainable params: 32,033
Non-trainable params: 0
_________________________________________________________________
None

**Matiriks Evaluasi model-2**
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10





----**Modul 3**----
- Jumlah splititng data set yaitu 
Data COVID19 Train : 460
Data COVID19 Validation : 10
Data COVID19 Test : 106
Data NORMAL Train : 1341
Data NORMAL Validation : 8
Data NORMAL Test : 234
Data PNEUMONIA Train : 3875
Data PNEUMONIA Validation : 8
Data PNEUMONIA Test : 390
Data TURBERCULOSIS Train : 650
Data TURBERCULOSIS Validation : 12
Data TURBERCULOSIS Test : 41
presentase spilt data train 80%
Presentaase split data test 19%
Presentase split data val 1%
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

batch_size = 32
augmen_gen = ImageDataGenerator(rescale=1 / 255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d (Conv2D)              (None, 200, 200, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 100, 100, 32)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 64)      18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 50, 50, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 50, 50, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 25, 25, 128)       0         
_________________________________________________________________
dropout (Dropout)            (None, 25, 25, 128)       0         
_________________________________________________________________
batch_normalization (BatchNo (None, 25, 25, 128)       512       
_________________________________________________________________
flatten (Flatten)            (None, 80000)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               10240128  
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 516       
=================================================================
Total params: 10,334,404
Trainable params: 10,334,148
Non-trainable params: 256
_________________________________________________________________
None
**Matiriks Evaluasi model-1**
              precision    recall  f1-score   support

           0       0.46      0.79      0.58       106
           1       0.72      0.35      0.47       234
           2       0.71      0.79      0.75       390
           3       0.60      0.59      0.59        41

    accuracy                           0.65       771
   macro avg       0.62      0.63      0.60       771
weighted avg       0.67      0.65      0.64       771



**loss model 1**
![image](https://user-images.githubusercontent.com/50208514/147497793-1acf0580-2957-46d8-989a-b37dda0b0cf5.png)
**Accuracy model 1**
![image](https://user-images.githubusercontent.com/50208514/147497822-259e6b38-9d8d-4ce7-835b-5d4615821c59.png)

**Model 2**
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d (Conv2D)              (None, 200, 200, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 100, 100, 32)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 64)      18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 50, 50, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 50, 50, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 25, 25, 128)       0         
_________________________________________________________________
dropout (Dropout)            (None, 25, 25, 128)       0         
_________________________________________________________________
batch_normalization (BatchNo (None, 25, 25, 128)       512       
_________________________________________________________________
flatten (Flatten)            (None, 80000)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               10240128  
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 516       
=================================================================
Total params: 10,334,404
Trainable params: 10,334,148
Non-trainable params: 256
_________________________________________________________________
None
**Matiriks Evaluasi model-2**
              precision    recall  f1-score   support

           0       0.88      0.72      0.79       106
           1       0.88      0.58      0.70       234
           2       0.75      0.95      0.84       390
           3       0.83      0.71      0.76        41

    accuracy                           0.79       771
   macro avg       0.83      0.74      0.77       771
weighted avg       0.81      0.79      0.79       771

**Loss model 2**
![image](https://user-images.githubusercontent.com/50208514/147497895-df3b60db-5e64-4fd3-b6fd-e31dd97b79ce.png)
**Accuracy model 2**
![image](https://user-images.githubusercontent.com/50208514/147497916-c8feaac3-8688-4bde-bc88-01ad09ec6e53.png)

**Model 3**
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param    
=================================================================
conv2d_6 (Conv2D)            (None, 200, 200, 32)      896       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 100, 100, 32)      0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 100, 100, 64)      18496     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 50, 50, 64)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 50, 50, 128)       73856     
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 25, 25, 128)       0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 25, 25, 128)       0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 25, 25, 128)       512       
_________________________________________________________________
flatten_2 (Flatten)          (None, 80000)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               10240128  
_________________________________________________________________
dropout_5 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 4)                 516       
=================================================================
Total params: 10,334,404
Trainable params: 10,334,148
Non-trainable params: 256
_________________________________________________________________
None
**Matiriks Evaluasi model-3**
              precision    recall  f1-score   support

           0       0.50      0.01      0.02       106
           1       0.67      0.01      0.02       234
           2       0.51      0.99      0.68       390
           3       0.80      0.20      0.31        41

    accuracy                           0.52       771
   macro avg       0.62      0.30      0.26       771
weighted avg       0.57      0.52      0.37       771

**Loss model 3**
![image](https://user-images.githubusercontent.com/50208514/147498001-d12e03e1-5e5e-48a8-9115-ac4c6688a876.png)
**Accuracy model 3**
![image](https://user-images.githubusercontent.com/50208514/147498014-bfffe7fb-e9d4-469d-9a16-53e1718eef55.png)

**Model saving**
import tensorflow as tf
model2.save('model3.h5')
new_model2 = tf.keras.models.load_model('model3.h5')

-Show the model architecture
new_model2.summary()





----**Modul 4**------
- Jumlah splititng data set yaitu 
Data COVID19 Train : 460
Data COVID19 Validation : 10
Data COVID19 Test : 106
Data NORMAL Train : 1341
Data NORMAL Validation : 8
Data NORMAL Test : 234
Data PNEUMONIA Train : 3875
Data PNEUMONIA Validation : 8
Data PNEUMONIA Test : 390
Data TURBERCULOSIS Train : 650
Data TURBERCULOSIS Validation : 12
Data TURBERCULOSIS Test : 41
presentase spilt data train 80%
Presentaase split data test 19%
Presentase split data val 1%

-Preprocessing data yang digunakan :
batch_size = 32
augmen_gen = ImageDataGenerator(rescale=1 / 255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
train_aug = augmen_gen.flow_from_directory(
    '/tmp/chest-xray-pneumoniacovid19tuberculosis/train',
    class_mode = 'categorical',
    shuffle=True,
    target_size = (250,250),
    batch_size=batch_size,
    color_mode ='rgb'

-Model yang digunakan sequential dengan menggunakan beberapa layers yaitu InputLayer, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
-Model 1
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 conv2d (Conv2D)             (None, 250, 250, 64)      1792      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 125, 125, 64)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 128)     73856     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 63, 63, 128)      0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 63, 63, 256)       295168    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 32, 32, 256)      0         
 2D)                                                             
                                                                 
 dropout (Dropout)           (None, 32, 32, 256)       0         
                                                                 
 batch_normalization (BatchN  (None, 32, 32, 256)      1024      
 ormalization)                                                   
                                                                 
 flatten (Flatten)           (None, 262144)            0         
                                                                 
 dense (Dense)               (None, 128)               33554560  
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 4)                 516       
                                                                 
=================================================================
Total params: 33,926,916
Trainable params: 33,926,404
Non-trainable params: 512
_________________________________________________________________
None

**Matiriks Evaluasi model-1**
precision    recall  f1-score   support

           0       0.86      0.29      0.44       106
           1       0.90      0.36      0.52       234
           2       0.73      0.93      0.82       390
           3       0.28      0.98      0.43        41

    accuracy                           0.67       771
   macro avg       0.69      0.64      0.55       771
weighted avg       0.78      0.67      0.65       771

**Loss Model-1**
![download](https://user-images.githubusercontent.com/50208514/143838358-4ecf38b9-0d3a-4829-9017-462076bba360.png)

**Accuracy Model-1**
![Accuracy](https://user-images.githubusercontent.com/50208514/143838544-a8965574-ef7a-4389-8dfc-6687506ce179.png)

--------------------------------------------------------------------------------------------------------------
**-Model 2**
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 conv2d_3 (Conv2D)           (None, 250, 250, 32)      896       
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 125, 125, 32)     0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 125, 125, 64)      18496     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 63, 63, 64)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 63, 63, 128)       73856     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 32, 32, 128)      0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 32, 32, 128)       0         
                                                                 
 batch_normalization_1 (Batc  (None, 32, 32, 128)      512       
 hNormalization)                                                 
                                                                 
 flatten_1 (Flatten)         (None, 131072)            0         
                                                                 
 dense_2 (Dense)             (None, 128)               16777344  
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense_3 (Dense)             (None, 4)                 516       
                                                                 
=================================================================
Total params: 16,871,620
Trainable params: 16,871,364
Non-trainable params: 256
_________________________________________________________________
None

**Matiriks Evaluasi model- 2**
precision    recall  f1-score   support

           0       0.86      0.67      0.75       106
           1       0.91      0.46      0.61       234
           2       0.73      0.97      0.84       390
           3       0.63      0.80      0.71        41

    accuracy                           0.77       771
   macro avg       0.78      0.73      0.73       771
weighted avg       0.80      0.77      0.75       771

**Loss Model-2**
![loss(2)](https://user-images.githubusercontent.com/50208514/143838864-b093db8a-4142-4700-be5f-d9ec9ce52985.png)

**Accuracy Model-2**
![Accuracy(2)](https://user-images.githubusercontent.com/50208514/143838898-16123232-f9ea-4c5a-bc42-7ad47bc36180.png)




-----**Modul 5**---
- Jumlah splititng data set yaitu 
Data COVID19 Train : 460
Data COVID19 Validation : 10
Data COVID19 Test : 106
Data NORMAL Train : 1341
Data NORMAL Validation : 8
Data NORMAL Test : 234
Data PNEUMONIA Train : 3875
Data PNEUMONIA Validation : 8
Data PNEUMONIA Test : 390
Data TURBERCULOSIS Train : 650
Data TURBERCULOSIS Validation : 12
Data TURBERCULOSIS Test : 41
presentase spilt data train 80%
Presentaase split data test 19%
Presentase split data val 1%

-Preprocessing data yang digunakan :
batch_size=10

augmen_gen = ImageDataGenerator(rescale=1/255,rotation_range=30, width_shift_range=0.1,
                        height_shift_range=0.1, shear_range=0.2, 
                        zoom_range=0.8, horizontal_flip=True,
                        fill_mode="nearest")

train_aug = augmen_gen.flow_from_directory(
    '/tmp/chest-xray-pneumoniacovid19tuberculosis/train',
    class_mode = 'categorical',
    shuffle=True,
    target_size = (50,50),
    batch_size=batch_size,
    color_mode ='rgb'
)

test_aug = augmen_gen.flow_from_directory(
    '/tmp/chest-xray-pneumoniacovid19tuberculosis/test',
    class_mode = 'categorical',
    shuffle=False,
    target_size = (50,50),
    batch_size=batch_size,
    color_mode ='rgb',
)

-Model pretrained yang digunakan VGG19 dengan menggunakan beberapa layers yaitu InputLayer, Dense, Conv2D, MaxPool2D, Flatten
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 input_1 (InputLayer)        [(None, 50, 50, 3)]       0         
                                                                 
 block1_conv1 (Conv2D)       (None, 50, 50, 64)        1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 50, 50, 64)        36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 25, 25, 64)        0         
                                                                 
 block2_conv1 (Conv2D)       (None, 25, 25, 128)       73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 25, 25, 128)       147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 12, 12, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 12, 12, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 12, 12, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 12, 12, 256)       590080    
                                                                 
 block3_conv4 (Conv2D)       (None, 12, 12, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 6, 6, 256)         0         
                                                                 
 block4_conv1 (Conv2D)       (None, 6, 6, 512)         1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
 block4_conv4 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
 block5_conv1 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                 
 block5_conv4 (Conv2D)       (None, 3, 3, 512)         2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0         
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 1000)              513000    
                                                                 
 dense_1 (Dense)             (None, 4)                 4004      
                                                                 
=================================================================
Total params: 20,541,388
Trainable params: 517,004
Non-trainable params: 20,024,384
_________________________________________________________________

**Matrik Evaluasi model 1**:
precision    recall  f1-score   support

           0       0.76      0.49      0.60       106
           1       0.90      0.48      0.63       234
           2       0.72      0.97      0.82       390
           3       0.63      0.78      0.70        41

    accuracy                           0.74       771
   macro avg       0.75      0.68      0.69       771
weighted avg       0.77      0.74      0.73       771

**Loss Model 1**: ![image](https://user-images.githubusercontent.com/50207475/147219652-096fcd0d-db27-4f8d-8faf-32233189e455.png)
**Accuracy Model 1**:![image](https://user-images.githubusercontent.com/50207475/147219688-6b15a254-4291-412d-b00c-b44cd4b9a588.png)

**Model 2 : Menambahkan DenseNet169 **
Matriks Evaluasi :
              precision    recall  f1-score   support

           0       0.75      0.52      0.61       106
           1       0.74      0.59      0.66       234
           2       0.78      0.88      0.83       390
           3       0.50      0.88      0.64        41

    accuracy                           0.74       771
   macro avg       0.69      0.72      0.68       771
weighted avg       0.75      0.74      0.74       771

**Loss Model 2** : ![image](https://user-images.githubusercontent.com/50207475/147220000-a50f8230-2fa6-4a8d-a088-acc162d2437d.png)
**Accuracy Model 1** : ![image](https://user-images.githubusercontent.com/50207475/147220017-1181c26e-057a-46bc-b850-a194e6bc125d.png)

(to generate a SavedModel) 
tf.saved_model.save(model, "saved_model_keras_dir")



--------**Modul 6**---------- 
Menggunakan 2 Training Model yaitu Pretrained model VGG16 dan Model manual Sequintial
Link Heroku : https://mod6.herokuapp.com/
