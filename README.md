# Tutorial

## Overview
+ Tensorflow 사이트 내 Image Classification Tutorial과 함께, Data Augmentation, Transfer Learning, Keras Guide 등의 Advanced 한 내용에 대한 이해.
+ 진행할 Kaggle Competition을 위한 Baseline 코드 작성.

## Description
+ **Tutorial.ipynb :** Jupyter notebook을 이용하여 Tensorflow 사이트 내 Image Classification에 대한 Tutorial로 Basic CNN model을 이용하여 Flower 데이터를 학습하고 Overfitting을 피하기 위한 Data Augmentation 진행.    
+ **Tutorial file :** 코드를 이해하기 위해 Jupyter notebook을 이용하여 한줄씩 작성하였다면 모델 학습 과정을 기능별로 묶는 Modular Programming을 진행하고, 여러 변수들을 용이하게 조절하기 위한 yaml 파일 생성. 각 Module에 대한 Description은 아래 같음.
  + hyperparameter.yaml: 여러 변수 값들의 용이한 조절을 위해 변수 값들을 모아놓은 yaml 파일.
  + data_generator.py: Train Data를 불러와 .
  + data_preprocessor.py: .
  + data_augmentation.py: Overfitting을 방지하기 위해 데이터를 Augmentate.
  + models.py: 학습을 위한 Basic Model과 Transfer Learning을 위해 Pretrained Model인 VGG16 모델을 생성 .
  + train_class.py:
  + train_func.py:
  + validate.py: 학습된 모델의 성능을 평가.
  + main.py: 생성한 모든 Module을 불러와 데이터를 이용하여 모델을 학습시킴.

* * *

# Kaggle Competition
## Overview
**Competition:** Cassava Leaf Diease Classification <br>
**Objective:** To classify each cassava image into four disease categories or a fifth category indicating a healthy leaf

## Description

## Results & Improvements
