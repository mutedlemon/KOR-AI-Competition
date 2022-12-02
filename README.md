# 2022 한국어 AI 경진대회
* 2022 한국어 AI 경진대회의 음성인식 성능평가 부문에서 장려상(NIA원장상)을 받은 코드입니다.
* 대회 링크: https://competition.aihub.or.kr/2022


## Contributor
* [한지원](https://github.com/mutedlemon)
* [홍지우](https://github.com/jiwooya1000)


## 모델 설명
* Deepspeech2: 
    * t1-cmd(명령어), t1-dial(방언발화) 데이터셋에 사용
    * 김수환 님께서 개발해 공개하신 kospeech (https://github.com/sooftware/kospeech) 를 기반으로 작성하고, NSML 플랫폼에서 사용 가능한 형태로 수정하였습니다.
* Conformer: 
    * t1-free(자유대화) 데이터셋에 사용
    * [Conformer: Convolution-augmented Transformer for Speech Recognition](https://www.isca-speech.org/archive_v0/Interspeech_2020/pdfs/3015.pdf)을 참고해서 작성하였습니다.


## 본선 데이터셋

### Dataset Name

`TRACK1-1 - 명령어` : `t1-cmd-final`  
`TRACK1-2 - 자유대화` : `t1-free-final`  
`TRACK1-3 - 방언발화` : `t1-dial-final`  

`rootpath = nsml.DATSET_PATH`
### Train Dataset

`DATASET_PATH/train/train_data/`  
- `train_data (헤더가 없는 pcm 형식)`
  - 파일명: idx000000 ~ 
  - PCM 샘플링 주파수: 16000Hz
  - Mono Channel


### Train Label

`DATASET_PATH/train/train_label`
  - `train_label (csv 형식)`
    - columns - `["filename", "text"]`
    - `filename` - train_data 폴더에 존재하는 파일명 (ex. idx000000)
    - `text` - train_data 폴더에 존재하는 파일의 음성 전사 Text 정보 (ex. 인공지능 훈민정음에 꽃 피우다)
    
### 참고사항
- 평가에 사용되는 데이터는 **철자 전사**를 따르고 있습니다.
-원천데이터에 포함된 개인정보는 라벨링시 익명처리 등 비식별화를 위해 아래와 같이 마스킹 처리되었습니다.   
   -  이름    :  `&name&  `  
   -  상호명    :    `&company-name&`  
   -  주민등록번호   :  `&social-security-num&`   
   -  카드번호     :   `&card-num&`  
   -  주소     :   `&address&`  
   -  전화번호    :    `&tel-num&`  
   -  정당명     :   `&party-name&`  
   
 >  자세한 정보는 [AI-HUB](https://www.aihub.or.kr/)의 해당 데이터에 대한 **구축 가이드 및 데이터 설명서**를 참고하실 수 있습니다.



## Code
- `main.py` : 실행파일
- `setup.py`: 환경설정(Base Docker Image, Python libraries)
- `nsml_package.txt`: packages(by apt or yum)
- `modules`
  - `audio` : 오디오 모듈(parsing)
  - `data.py` : 데이터 로더
  - `inference.py`: 인퍼런싱
  - `metrics.py` : 평가지표 관련(CER)
  - `model.py`: 모델 빌드 관련(DeepSpeech2)
  - `preprocess.py`: 전처리(라벨/transcripts 제작)
  - `trainer.py`: 학습 관련
  - `utils.py` : 기타 설정 및 필요 함수
  - `vocab.py` : Vocabulary Class 파일

### 실행 방법
```bash
# 명칭이 't1-cmd'인 데이터셋을 사용해 세션 실행하기
$ nsml run -d t1-cmd
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nsml run -d t1-cmd -e anotherfile.py
$ nsml run -d t1-cmd -e [파일명]
# 2GPU와 16CPU, 160GB 메모리를 사용하여 세션 실행하기   
$ nsml run -d t1-cmd -g 2 -c 16 --memory 160G  

## 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml logs -f [세션명]

## 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml model ls [세션명]

## 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nsml submit -t [세션명] [모델_checkpoint_번호]

## 모델 제출하기
## 제출 후 리더보드에서 점수 확인 가능
nsml submit [세션명] [모델_checkpoint_번호]
```
