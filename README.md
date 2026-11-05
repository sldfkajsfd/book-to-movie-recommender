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

MiniLM 모델 출처는 sentence-transformers/all-MiniLM-L6-v2이며 revision.txt에 고정 버전을 보존했다. CPU는 ONNX Runtime의 양자화 모델을 사용한다. 선택 가능한 gpu는 OpenVINO Intel GPU 경로이며 NVIDIA/Colab GPU 경로는 아직 구현하지 않았다. Intel GPU 실행에는 openvino==2026.4.1을 별도로 설치한다.

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

Colab에서는 프로젝트의 소스와 `data/`를 런타임에 준비하고 위 명령을 실행한다. TF-IDF, 책 CF, 추천 점수와 평가 지표 계산은 CPU에서 수행한다. Colab의 시스템 RAM이 이 단계의 메모리 한도를 정한다. MiniLM을 실행할 때는 `models/minilm/`과 해당 실행 패키지도 필요하다. 현재 `--device gpu`는 Intel OpenVINO 전용이다.

## 저장되는 것

- results/\*\_recommendations.csv: 모델별 추천
- results/metrics.json: 이번 실행의 평가 수치
- results/cache/: 재사용할 벡터·사전·CF와 캐시 검증 정보

원본 데이터는 수정하지 않는다. 실행 도중 중단된 벡터 생성은 해당 단계를 다시 계산한다.

## 모델 조건

20만 TF-IDF, unigram+bigram, 책 전용 사전·IDF, 4~5점 책 동일 비중, 코사인 Top 10. 20만 사전으로 변환한 벡터가 0이 아닌 책과 영화를 공통 작품 목록으로 사용한다. CF는 이 책 목록에서 최대 100명 이웃을 사용한다. 기본값은 미평가 책 최대 20권과 직접 취향 대 CF 취향 50:50이다. CF가 불가능하면 직접 취향을 그대로 사용한다. MiniLM도 공통 작품 목록을 사용한다. 영화 평점은 평가만 담당한다.

## 검증 범위

소규모 입력으로 실행을 검증했다. 최종 평가 지표와 후보 수는 전체 데이터 실행 후 확인한다.

## 폴더 정리

과거 실험과 임시 파일은 보조자료/에 모았다. 실행에는 data/, models/, src/, main.py, stopwords.json이 필요하다. 제출 ZIP에는 코드와 설치 안내, 불용어 설정을 담으며 원본 데이터와 모델 파일은 별도로 준비한다.
