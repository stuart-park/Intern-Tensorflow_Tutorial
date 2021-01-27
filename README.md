# Tutorial

## Overview
+ Tensorflow 사이트 내 Image Classification Tutorial과 함께, Data Augmentation, Transfer Learning, Keras Guide 등의 Advanced 한 내용에 대한 이해.
+ 진행할 Kaggle Competition을 위한 Baseline 코드 작성.

## Description
+ **Tutorial.ipynb :** Jupyter Notebook을 이용하여 Tensorflow 사이트 내 Image Classification에 대한 Tutorial 진행. 해당 Tutorial에서는 Basic CNN model을 이용하여 Flower 데이터를 학습하고 Overfitting을 피하기 위한 Data Augmentation 진행.    
+ **Tutorial file :** VSCode를 이용하여 Jupyter Notebook에 작성한 코드를 기능별로 묶는 Modular Programming을 진행하고, 여러 변수들을 용이하게 조절하기 위한 yaml 파일 생성, VGG16을 이용한 Transfer Learning, `GradientTape()`를 이용한 `fit()` API의 구현 등을 진행. 
  + hyperparameter.yaml: Hyperparameter 값 조절.
  + data_generator.py: Data를 불러와 Train Data와 Validation Data로 split한 후, `map()` API를 이용하여 (Image, Label) 쌍으로 변환. 
  + data_preprocessor.py: [0, 255] 범위의 값들을 신경망 학습을 위해 [0, 1] 범위로 Rescaling 해주고, Performance 향상을 위 Data configuration 진행.
  + data_augmentation.py: Overfitting 방지를 위해 Augmentation 과정을 진행.
  + models.py: Basic Model과, Transfer Learning을 위해 Pretrained Model인 VGG16 모델을 생성.
  + train_class.py: Keras내 `fit()` API를 이용하지 않고 `GradientTape()`를 이용하여 한 Batch당 Loss를 확인할 수 있도록 구현.
  + train_func.py: 위와 동일하게 기능을 하지만, class가 아닌 function으로 구현.
  + validate.py: 1 epoch동안 학습한 모델을 Validate.
  + main.py: 위 Module을 모두 Import하여 Image Classification을 진행.

* * *

# Kaggle Competition
## Overview
+ **Competition :** Cassava Leaf Diease Classification
+ **Objective :** Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf
+ **Evaluation :** Categorization accuracy

## Description

## Results & Improvements
