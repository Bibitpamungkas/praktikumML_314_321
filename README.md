# praktikumML_314_321
Tugas Kelompok Praktikum Machine Learning

Overview jurnal :
-Data yang digunakan pada jurnal yang berjudul Unveiling COVID-19 from CHEST X-Ray with
Deep Learning: A Hurdles Race with Small Data, yaitu data Covid-19 rontgen dada di salah satu rumah sakit darurat utama tepatnya di daerah italia utara
-preprocessing data yang dilakukan dataset pada jurnal ini memiliki tujuan untuk menghilangkan bias dalam data. Yang pertama menggunakan ekualisasi histogram guna menjamin dinamika gambar menjadi seragam di dalam data.kemudian melakukan segmentasi pada paru-paru guna mengelompokkan gambar dengan gambar yang ditampilkan hanya paru-paru saja, kemudian diburamkan pada tepian dengan menggunakan radisu 3 pixel guna mengatasi tepi yang tajam.
-Model yang digunakan pada jurnal rujukan yaitu ResNet-18, ResNet-50, COVID-Net, DenseNet-121

Overview dataset :
-Link Sumber dataset :https://www.kaggle.com/jtiptj/chest-xray-pneumoniacovid19tuberculosis
-Jumlah data gambar keseluruhan 7135 image, dan dibagi menjadi 3 yaitu train, test, val. jumlah nilai di dalamnya yaitu pada label train 6326 images, label test 771 images, dan label val 38 images.
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
 Layer (type)                Output Shape              Param #   
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

Matiriks Evaluasi model-1
precision    recall  f1-score   support

           0       0.86      0.29      0.44       106
           1       0.90      0.36      0.52       234
           2       0.73      0.93      0.82       390
           3       0.28      0.98      0.43        41

    accuracy                           0.67       771
   macro avg       0.69      0.64      0.55       771
weighted avg       0.78      0.67      0.65       771

Loss Model-1
![download](https://user-images.githubusercontent.com/50208514/143838358-4ecf38b9-0d3a-4829-9017-462076bba360.png)

Accuracy Model-1
![Accuracy](https://user-images.githubusercontent.com/50208514/143838544-a8965574-ef7a-4389-8dfc-6687506ce179.png)

--------------------------------------------------------------------------------------------------------------
-Model 2
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
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

Matiriks Evaluasi model- 2
precision    recall  f1-score   support

           0       0.86      0.67      0.75       106
           1       0.91      0.46      0.61       234
           2       0.73      0.97      0.84       390
           3       0.63      0.80      0.71        41

    accuracy                           0.77       771
   macro avg       0.78      0.73      0.73       771
weighted avg       0.80      0.77      0.75       771

Loss Model-2
![loss(2)](https://user-images.githubusercontent.com/50208514/143838864-b093db8a-4142-4700-be5f-d9ec9ce52985.png)

Accuracy Model-2
![Accuracy(2)](https://user-images.githubusercontent.com/50208514/143838898-16123232-f9ea-4c5a-bc42-7ad47bc36180.png)

