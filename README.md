# Tutorial

## Overview
+ Tensorflow 사이트 내 Image Classification Tutorial과 함께, Data Augmentation, Transfer Learning, Keras Guide 등의 Advanced 한 내용에 대한 이해.
+ 진행할 Kaggle Competition을 위한 Baseline 코드 작성.

## Description
+ **Tutorial.ipynb :** Jupyter Notebook을 이용하여 Tensorflow 사이트 내 Image Classification에 대한 Tutorial 진행. 본 Tutorial에서는 Basic CNN model을 이용하여 Flower 데이터를 학습하고 Overfitting을 피하기 위한 Data Augmentation 진행.    
+ **Tutorial file :** VSCode를 이용하여 Jupyter Notebook에서 작성한 코드를 기능별로 묶는 Modular Programming을 진행하고, 여러 변수들을 용이하게 조절하기 위한 yaml 파일 생성, VGG16을 이용한 Transfer Learning, `GradientTape()`를 이용한 `fit()` API의 구현 등을 진행. 
  + hyperparameter.yaml : Hyperparameter 값 조절.
  + data_generator.py : 학습할 Data들의 File Path에 대한 Dataset을 만들고, Train Data와 Validation Data로 split 시킴. 
  + data_preprocessor.py : `map()` API를 이용하여 File Path에 대한 데이터를 (Image, Label) 쌍으로 변환, Image의 Rescaling, Performance 향상을 위한 Data configuration 진행.
  + data_augmentation.py : Overfitting 방지를 위한 Augmentation 과정 진행.
  + models.py : Basic Model, Transfer Learning을 위한 Pretrained Model인 VGG16 Model, 필요에 따라 구조가 비슷한 VGG16과 VGG19를 사용하기 위한 class 생성.
  + train_func.py : Keras내 `fit()` API를 이용하지 않고 `GradientTape()`를 이용하여 한 Batch당 Loss를 확인할 수 있도록 구현.
  + train_class.py : 위와 동일한 기능을 하지만, `@tf.function`의 class 내 작동 여부 확인을 위해 function이 아닌 class로 구현.
  + validate.py : 1 epoch 동안 학습한 모델을 Validate.
  + main.py : 위 Module을 모두 Import하여 Image Classification 과정을 진행.

* * *

# Kaggle Competition
## Overview
+ **Competition :** Cassava Leaf Diease Classification
+ **Objective :** Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf
+ **Evaluation :** Categorization accuracy

## Description

## Results & Improvements
