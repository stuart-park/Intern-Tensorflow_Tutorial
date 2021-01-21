# Tutorial

## Overview
Following tutorial guides on tensorflow in jupyter notebook and modular programming via vscode

## Description
+ Tutorial.ipynb: Code in image classification for tutorial. Used tf.data instead of from_directory. 
+ Tutorial file: Modulizing jupyter code into it's function. Create class by it's function and initialize it. Also I created a parser for yaml file to easily manipulate some hyper parameters in model. Also I tried 3 kind of model traning. One is using 'fit()' API in keras. The other two is using customized fit, each is made with a class and a function. There was no big difference in training time, but 'fit()' API was more easy to use. 

* * *

# Kaggle Competition
## Overview
**Competition:** Cassava Leaf Diease Classification <br>
**Objective:** To classify each cassava image into four disease categories or a fifth category indicating a healthy leaf

## Description

## Daily Log
+ 10 / 20 (WED) : 
+ 10 / 21 (THU) : EfficientNetB3에서의 HyperParameter(resolution, lr, batch_size)를 바꿔가며 학습을 하였지만 val_acc가 0.78을 넘지 못함. 원인은 base_model을 freeze 시켜놓고 classifier부분만 학습을 시켜 Transfer Learning을 진행하였기 때문이었음. base model를 unfreeze시키고 모든 layer의 weight를 다시 학습(lr=1e-3, epochs=10)시켜 Fine-Tuning을 진행한 결과 val_acc가 0.81 정도로 향상되었음. Fine-Tuning시 learning rate을 낮게 잡지 않으면 빠르게 overfitting이 될 수 있기 때문에 lr을 낮게 잡고 sceduling을 시켜야겠다. 
