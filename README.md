# 책 취향 기반 영화 추천

## 코드를 읽는 순서

| 파일                  | 역할                               |
| --------------------- | ---------------------------------- |
| main.py               | 전체 실행 흐름                     |
| src/data.py           | 파일 읽기, ID 연결, 동점 순위 처리 |
| src/features.py       | 책 전용 TF-IDF와 MiniLM 임베딩     |
| src/content_based.py | 긍정 책 평균과 영화 Top 10         |
| src/hybrid.py         | 책 기반 사용자 KNN과 취향 결합     |
| src/editions.py       | 영화 판본 통합                     |
| src/evaluation.py     | 영화 평점 기반 평가                |

실제 기능 파일은 main 포함 7개다. src/**init**.py는 빈 패키지 파일이다. 과거 결과나 개발 기록을 읽지 않는다.

## 설치와 데이터

Python 3.12에서 `python -m pip install -r requirements.txt`를 실행한다. 원본 CSV 3개·JSONL 2개는 data/에 둔다. 불용어는 stopwords.json, MiniLM 모델은 models/minilm/에 있다. 원시 데이터 전처리는 이 코드에 포함되지 않는다.

MiniLM은 sentence-transformers/all-MiniLM-L6-v2를 사용한다. 모델 버전은 models/minilm/revision.txt에서 확인할 수 있다. cpu는 기존 양자화 ONNX, cpu_fp32는 FP32 ONNX, gpu는 기존 Intel OpenVINO를 사용한다. cuda는 같은 FP32 ONNX를 NVIDIA CUDA로 실행한다. Intel GPU 실행에는 openvino==2026.4.1을 별도로 설치한다.

## 실행

```text
python -X utf8 main.py run --model tfidf
python -X utf8 main.py run --model minilm --device cpu
python -X utf8 main.py results
```

각 run은 Content-based와 Hybrid를 함께 실행한다. --model all은 네 모델을 실행한다. --data와 --output으로 경로를 지정할 수 있다. MiniLM만 실행해도 공통 비교 후보를 만들기 위한 TF-IDF와 CF를 먼저 준비한다. 기존 캐시가 있으면 재사용하며, 입력이나 코드가 달라지면 새 --output을 지정해야 한다.

## Hybrid 비교 실험

CF 비중과 추론 책 수를 비교하려면 한 번의 실행에 값을 함께 지정한다.

```text
python -X utf8 main.py run --model tfidf --cf-percent 50 20 --cf-books 20 50 --output results_ablation
```

Content-based 하나와 Hybrid 네 조합을 평가한다. 비중은 직접 취향과 CF 취향 중 CF가 차지하는 비율이며, 책 수는 최대값이다. 같은 이웃 점수와 책 순위를 사용해 두 책 수를 비교한다. `metrics.json`에 비중과 책 수가 함께 저장된다. MiniLM을 별도로 비교하려면 같은 명령에 `--model minilm --device cpu`를 사용한다.

## Google Colab에서 NVIDIA GPU로 실행

1. [colab_run.ipynb](colab_run.ipynb)를 Colab에서 열고 NVIDIA GPU 런타임을 선택한다.
2. Drive의 `book-to-movie-recommender/data/`에 입력 5개를 넣는다. 같은 프로젝트의 `models/minilm/`에 `tokenizer.json`과 FP32 `model.onnx`를 넣는다.
3. 노트북을 위에서 아래로 실행한다. GPU 환경 확인, 브랜치 가져오기, 가상환경 설치, Drive 연결, CPU/CUDA 표본 검증을 거쳐 네 모델을 실행한다. 실제 CUDA 연산과 표본 검증을 확인한 뒤 전체 실행이 시작된다.
4. `runs/ablation_200k_cf20_top50_cuda/`에서 추천 CSV와 `metrics.json`, `ranking_checks.json`, `run_stats.json`, `resource_stats.json`, `cuda_check.json`을 확인한다.

직접 실행할 때는 NVIDIA 환경의 별도 가상환경에 `requirements-colab.txt`를 설치하고 아래 명령을 사용한다.

```text
python main.py run --model all --device cuda --cf-percent 20 --cf-books 50 --batch-size 32 --output results_cuda
```

Colab 설치 목록은 ONNX Runtime GPU 1.23.2, CUDA 12 계열과 cuDNN 9를 사용한다. CUDA 런타임과 cuBLAS는 12.8 이상을 설치한다. 노트북은 Python 3.10~3.13을 지원 대상으로 확인한다. 실제 드라이버와 패키지 버전은 `runtime_environment.json`에 기록하고, 설치 호환성은 CUDA 세션 초기화와 실제 연산으로 확인한다. [ONNX Runtime 호환성 표](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) · [PyPI 배포 파일](https://pypi.org/project/onnxruntime-gpu/1.23.2/) · [Colab 런타임 버전](https://research.google.com/colaboratory/runtime-version-faq.html)

MiniLM의 ONNX 추론만 CUDA로 가속한다. 토큰화, 평균 풀링·정규화, TF-IDF·CF, 추천·평가는 CPU에서 수행한다. GPU 메모리가 부족하면 `BATCH_SIZE`를 16 또는 8로 낮추고 `RUN_NAME`을 새로 지정한다. 무료 GPU 배정과 실행 시간은 가용성에 따라 달라진다. [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)

입력은 Drive에서 Colab 로컬 디스크로 복사하고, 완료된 추천·평가와 캐시는 실행 중 Drive로 복사한다. 다음 세션에서 같은 실행 이름을 사용하면 완료된 캐시와 결과를 복구할 수 있다. 중단 당시 진행 중이던 단계는 다시 계산한다. 입력·코드·수치 라이브러리 또는 임베딩 조건이 달라지면 새 실행 이름을 쓴다.

`run_stats.json`에는 MiniLM 벡터 생성 시간과 전체 실행의 경과 시간이 기록된다. `resource_stats.json`은 실행 프로세스의 CPU 사용률·RSS, 시스템 RAM 및 NVIDIA 프로세스 메모리의 표본 최고값을 기록한다. CPU 사용률 100%는 코어 하나에 해당한다. 실제 최대값은 샘플 사이에 더 높을 수 있다. 노트북 마지막 셀에서 동일 FP32 모델의 전체 CPU 측정을 선택할 수 있다. 비교 시 캐시 적중 여부와 Drive 복사 시간이 포함됐는지 확인한다.

## 저장되는 것

- results/\*\_recommendations.csv: 모델별 추천
- results/metrics.json: 이번 실행의 평가 수치
- results/cache/: 재사용할 벡터·사전·CF와 캐시 검증 정보

원본 데이터는 수정하지 않는다. 실행 도중 중단된 벡터 생성은 해당 단계를 다시 계산한다.

## 모델 조건

20만 TF-IDF, unigram+bigram, 책 전용 사전·IDF, 4~5점 책 동일 비중, 코사인 Top 10. 20만 사전으로 변환한 벡터가 0이 아닌 책과 영화를 공통 작품 목록으로 사용한다. CF는 이 책 목록에서 최대 100명 이웃을 사용한다. 기본값은 미평가 책 최대 50권과 직접 취향 대 CF 취향 80:20이다. CF가 불가능하면 직접 취향을 그대로 사용한다. MiniLM도 공통 작품 목록을 사용한다. 영화 평점은 평가만 담당한다.

## 검증 범위

소규모 입력으로 실행을 검증했다. 최종 평가 지표와 후보 수는 전체 데이터 실행 후 확인한다.

## 폴더 정리

과거 실험과 임시 파일은 보조자료/에 모았다. 실행에는 data/, models/, src/, main.py, stopwords.json이 필요하다. 제출 ZIP에는 코드와 설치 안내, 불용어 설정을 담으며 원본 데이터와 모델 파일은 별도로 준비한다.
