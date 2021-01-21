# 개발일지
+ **10 / 15 (FRI)**  
Tutorial을 통해 생성한 baseline code 에서 pretrained model을 `VGG16`에서 `EfficientNetB3`로 변경하고 학습을 진행시켜지만 OOM(Out of Memory)에러가 발생
+ **10 / 18 (MON)** 
OOM 에러의 원인을 찾다 보니 파일 디렉토리를 통해 이미지를 읽으들어온뒤 shuffle의 buffer size를 학습 데이터 크기인 20000으로 정하였기 때문. 데이터 파일 디렉토리로만 이루어져 있으면 문제가 없었지만 directory를 통해 이미지를 불러와서 array로 만들었기 때문에 shuffle의 buffer size를 1000으로 수정하니 모델이 학습이 되었다. 이렇게 학습된 모델을 통해 submit을 하려고 보니 본 대회는 code competition이라 코드를 제출하였는데 notebook timeout 에러 발생
+ **10 / 19 (TUE)**
notebook timeout에러 발생 해결을 위해 Kaggle을 찾아보니 code Competition 같은 경우 training 부분과 inference 부분을 나누어 submit시 train에서 저장된 모델을 불러와 inference 부분을 통해 추론하는 코드를 제출. 하지만 또 notebook timeout 에러 발생. 알고보니 batch를 잘못주어 발생한 에러. 코드 수정후 다시 submit 했는데 이제는 결과 0점으로 나와버렸다.
+ **10 / 20 (WED)** 
0점 원인을 계속 찾다보니까 inference 부분에서 출력층의 activation function이 softmax이다 보니 `predict()`에서 argmax를 붙여줬는데 jupyter에서 찍어보니 각 사진에 대한 것이 아닌 전체 데이터셋에 대한 argmax가 나온것이었다. 결국 test data에서 이미지를 한장씩 불러와서 예측하도록 코드를 수정하였고 그 결과 0.76 정도가 나옴. 결과를 향상시키기 위해 EfficinetB3 모델에서 learning rate, epoch, resolution(img_size) 등의 hyper parameter를 수정하여 학습시켰지만 0.78을 모두 넘지 못하였음. 
+ **10 / 21 (THU)**
EfficientNetB3에서의 HyperParameter(resolution, lr, batch_size)를 바꿔가며 학습을 하였지만 val_acc가 0.78을 넘지 못함. 원인은 base_model을 freeze 시켜놓고 classifier부분만 학습을 시켜 Transfer Learning을 진행하였기 때문이었음. base model를 unfreeze시키고 모든 layer의 weight를 다시 학습(lr=1e-3, epochs=10)시켜 Fine-Tuning을 진행한 결과 val_acc가 0.81 정도로 향상되었음. Fine-Tuning시 learning rate을 낮게 잡지 않으면 빠르게 overfitting이 될 수 있기 때문에 lr을 낮게 잡고 sceduling을 시켜야겠다. 
+ **10 / 22 (FRI)** 
