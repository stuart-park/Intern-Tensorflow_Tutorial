# Tutorial

## Overview
+ Tensorflow 사이트 내 Image Classification Tutorial과 함께, Data Augmentation, Transfer Learning, Keras Guide 등의 Advanced 한 내용에 대한 이해.
+ 진행할 Kaggle Competition을 위한 Baseline 코드 작성.

## Description
+ **Tutorial.ipynb :** Jupyter Notebook을 이용하여 Tensorflow 사이트 내 Image Classification에 대한 Tutorial 진행. 해당 Tutorial에서는 Basic CNN model을 이용하여 Flower 데이터를 학습하고 Overfitting을 피하기 위한 Data Augmentation 진행.    
+ **Tutorial file :** VSCode를 이용하여 Jupyter Notebook에 작성한 코드를 기능별로 묶는 Modular Programming을 진행하고, 여러 변수들을 용이하게 조절하기 위한 yaml 파일 생성, VGG16을 이용한 Transfer Learning, `fit()` API의 customizing 등을 진행. 각 Module에 대한 기능은 아래 같음.
  + hyperparameter.yaml: Hyperparameter 값 조절.
  + data_generator.py: Data를 불러와 Train Data와 Validation Data로 split한 후, `map()` API를 이용하여 (Image, Label) 쌍으로 변환. 
  + data_preprocessor.py: [0, 255] 범위의 값들을 신경망 학습을 위해 Standardize 시키고, Performance 향상을 위해 Data configuration 진행
  + data_augmentation.py: Overfitting을 방지하기 위해 데이터를 Augmentate.
  + models.py: 학습을 위한 Basic Model과 Transfer Learning을 위해 Pretrained Model인 VGG16 모델을 생성.
  + train_class.py: Keras내 `fit()` API를 이용하지 않고 직접 class로 customize.
  + train_func.py: 위와 동일하게 직접 `fit()` API를 직접 customize 했지만 위와 다르게 function으로 customize.
  + validate.py: `fit()`내의 argument인 validation 부분을 customize한 것으로 학습된 모델의 성능을 평가.
  + main.py: 생성한 모든 Module을 불러와 데이터를 이용하여 모델을 학습시킴.

* * *

# Kaggle Competition
## Overview
![cassava](https://user-images.githubusercontent.com/53407163/105679077-4a543180-5f31-11eb-8e5a-a9401771de3c.JPG)
+ **Competition :** Cassava Leaf Diease Classification
+ **Objective :** Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf
+ **Evaluation :** Categorization accuracy

## Description

## Results & Improvements
