# 창의적통합설계2 INFINITT healthcare X-ray 팀
자유전공학부 김재원  
컴퓨터공학부 남재호  
컴퓨터공학부 박지나  

### how to install python library prerequisites
`pip install -r requirements.txt`

### how to run execution program
`cd demo_codes`  
`python execution.py`

### dataset schema
`cat structure.txt`

## demo codes

#### 200529_model.h5 <br/> 200529_model.json
keras로 만든 FCN 모델입니다. 실행 프로그램이 시작할 때 불러와서 사용합니다.

#### execution.py <br/> execution.pyw
실행 프로그램의 실행 코드입니다. .pyw 파일은 명령 프롬프트를 실행하지 않을 때 사용합니다.

#### display_metadata.py
<img src="https://user-images.githubusercontent.com/32262002/85721116-07787400-b72c-11ea-8418-689daa454c09.png" width="50%" height="50%">
input X-ray 영상의 정보를 불러오기 위한 코드입니다.

#### histogram_equalization.py
<img src="https://user-images.githubusercontent.com/32262002/85721362-40b0e400-b72c-11ea-9397-da94a3173fee.png" width="50%" height="50%">
histogram equalization을 실행하기 위한 코드입니다.

#### brightness_contrast_control.py
<img src="https://user-images.githubusercontent.com/32262002/85722678-7efad300-b72d-11ea-954e-712672a60efd.png" width="50%" height="50%">
영상의 밝기와 대비를 조절하기 위한 코드입니다.

#### windowing.py <br/> windowing.png
<img src="https://user-images.githubusercontent.com/32262002/85722970-c1bcab00-b72d-11ea-898c-c691ae3281c3.png" width="50%" height="50%">
영상의 windowing을 조절하기 위한 코드입니다.

#### canny_edge_detection.py
<img src="https://user-images.githubusercontent.com/32262002/85723870-9e463000-b72e-11ea-8e4e-277c16d558f3.png" width="50%" height="50%">
lung에 대해 canny edge detection으로 landmark를 찾기 위한 코드입니다.

#### black_white_windowing.py
<img src="https://user-images.githubusercontent.com/32262002/85723544-4e676900-b72e-11ea-917b-9cc7e211cbeb.png" width="50%" height="50%">
lung에 대해 windowing으로 landmark와 anatomical imaging range를 찾기 위한 코드입니다.

#### fcn.py
<img src="https://user-images.githubusercontent.com/32262002/85724487-36dcb000-b72f-11ea-9224-f35138911d78.png" width="50%" height="50%">
lung에 대해 FCN 모델을 이용하여 landmark와 anatomical imaging range를 찾기 위한 코드입니다.

#### skull_super_pixel_clustering.py
<img src="https://user-images.githubusercontent.com/32262002/85724814-891dd100-b72f-11ea-9f89-916e3eab24f0.png" width="50%" height="50%">
skull에 대해 super pixel clustering으로 landmark와 anatomical imaging range를 찾기 위한 코드입니다.

#### skull.py
<img src="https://user-images.githubusercontent.com/32262002/85725290-f16cb280-b72f-11ea-9fe5-c5cc37188f0a.png" width="50%" height="50%">
skull에 대해 windowing으로 anatomical imaging range를 찾기 위한 코드입니다.

#### plt.py
결과 출력을 위한 matplotlib.pyplot 코드를 공유하는 코드입니다.

#### make_dicom.py
head X-ray가 존재하지 않아 png, jpg 파일을 사용하여 DICOM 파일을 만들기 위한 코드입니다.

## lung ##

### fcn
#### └ predict_image
lung DICOM image에 대해 predict를 해 본 결과를 저장하는 폴더입니다.

#### └ test_result
anatomical imaging range를 찾기 위해 여러가지 방법으로 테스트해 본 코드입니다.

#### └ 200420_test256.txt ~ 200529_result.txt
4월부터 현재까지 여러 모델을 만들어 본 기록이고, `200529_model.h5`와 `200529_model.json`이 최신 버전입니다. 따라서 200420 ~ 200523 result, model은 중요하게 생각하지 않아도 됩니다. 200529_model을 어떻게 만들었는지는 200529_result.txt에 나와 있습니다. 먼저 450개의 train set, 113개의 val set, 141개의 test set으로 구성되었으며, 512 * 512 size의 image로 preprocessing된 input을 사용합니다. model의 구조는 5개의 (conv - maxpooling), 1개의 dense, 5개의 (upsampling - conv)로 구성됩니다. 전체 param의 개수는 2M 정도이고, epoch 100회에 대해 dice coefficient가 95% 이상 나왔습니다.

#### ├ preprocess_256.py <br/> └ preprocess_512.py
학습을 png image로 진행하였기 때문에 DICOM file을 불러오지는 않습니다. 불러온 이미지 파일을 256 * 256 또는 512 * 512 size의 numpy array로 바꾸어 저장합니다. 최신 버전은 512 * 512 size를 적용한 `preprocess_512.py`입니다.

#### ├ train_256.py <br/> ├ train_512.py <br/> └ train_512_v2.py
preprocess_256.py, preprocess_512.py에서 저장한 numpy array를 불러와 학습을 진행합니다. 학습에 사용되었던 주요 설정은 다음과 같습니다.  
optimizer = Adam  
metrics = [dice_coef]  
epochs = 100  
batch_size = 8  
callbacks = ReduceLROnPlateau  
learning rate = 0.2  
patience = 8  
`train_512.py`가 최신 버전이며, train_256.py는 이전해 시도해보았던 코드이고, train_512_v2.py는 시도해보았으나 실패한 코드입니다.

#### ├ predict_dicom.py <br/> └ predict.py
anatomical imaging range를 잘 찾아내는지 검사하기 위해 SIIM dataset을 이용하여 테스트하는 코드입니다. DICOM 파일과 bounding box metadata를 불러와서 landmark 추출, anatomical imaging range 계산을 마친 후 정답과 결과의 차이를 저장합니다. 그 결과의 차이로 k-fold validation을 수행합니다. `predict_dicom.py`가 최신 버전입니다.

#### └ kfold.py
predict_dicom.py에서 예측한 결과를 토대로 10-fold validation을 수행합니다. 정확도는 +- 2cm 오차 안으로 들어오는 경우 정답이라고 하였고, 벗어나는 경우 오답이라고 정의한 metric입니다.

#### ├ mask_to_img.py <br/> ├ merge_leftMask_and_rightMask.py <br/> └ xray_to_img.py
dataset을 만들기 위해 image를 전처리하는 코드입니다. 크게 중요하지 않습니다.

