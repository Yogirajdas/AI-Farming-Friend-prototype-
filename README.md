# AI-Farming-Friend-prototype
Here a prototype of machine learning project called "AI-Farming-Friend" is given also some code implementation on exploratory data analysis along with model building is done.
Target is to create a AI/ML service/product prototype which will help farmers in business growth.
Here work is done based on kaggle dataset. The actual implementation will require more detailed dataset containing various more features of market situation, weather, time, region, etc. Also the performace of model will be required to near 99% for each case.
For that correct seed(crop), fertilizer selection along with monitoring crop to ensure health and growth (disease detection and weed detection) is necessary. 
Here EDA is done on crop selection data, also some ml models are tested for the same. CNN Model is also build for disease detection part and code is included. In the similar way For fertillizer and weed detection we can get results.

Some results of implementation part:

•	K nearest neighbors

knn_train_accuracy = 0.9886363636363636
knn_test_accuracy = 0.975 

•	Decision Tree

Training accuracy = 0.8818181818181818
Testing accuracy = 0.9

•	Random forest

Training accuracy = 1.0
Testing accuracy = 0.990909090909091

•	Naïve Bayes classifier

Training accuracy = 0.9960227272727272
Testing accuracy = 0.990909090909091

•	XGBoost

Training accuracy = 1.0
Testing accuracy = 0.9931818181818182

Here is the model layers of densenet121 for disease classification
Model: "leaf_disease_model"

Layer (type)                 Output Shape              Param #   

input_28 (InputLayer)        [(None, 224, 224, 3)]     0         

densenet121 (Functional)     (None, 1000)              8062504   

dropout_11 (Dropout)         (None, 1000)              0         

dense_10 (Dense)             (None, 38)                38038     

Total params: 8,100,542
Trainable params: 38,038
Non-trainable params: 8,062,504

Here InceptionV3 or ResNet can also be used for better performance of prediction. 
Laterwards web interface can be built with angular, express or Django, flask framework to deploy application.





  

