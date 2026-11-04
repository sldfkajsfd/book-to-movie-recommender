# 책 취향 기반 영화 추천

## 코드를 읽는 순서

| 파일                  | 역할                               |
| --------------------- | ---------------------------------- |
| main.py               | 전체 실행 흐름                     |
| src/data.py           | 파일 읽기, ID 연결, 동점 순위 처리 |
| src/features.py       | 책 전용 TF-IDF와 MiniLM 임베딩     |
| src/recommendation.py | 긍정 책 평균과 영화 Top 10         |
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

기존 보고서의 측정값만 보려면 다음 명령을 사용한다.

```text
python -X utf8 main.py results --reference
```

## 저장되는 것

- results/\*\_recommendations.csv: 모델별 추천
- results/metrics.json: 이번 실행의 평가 수치
- results/cache/: 재사용할 벡터·사전·CF와 캐시 검증 정보
- reference_metrics.json: 정리 이전 보고서의 평가 수치

원본 데이터는 수정하지 않는다. 실행 도중 중단된 벡터 생성은 해당 단계를 다시 계산한다. 과거 결과를 새 실행에서 측정한 것처럼 표시하지 않는다.

## 모델 조건

20만 TF-IDF, unigram+bigram, 책 전용 사전·IDF, 4~5점 책 동일 비중, 코사인 Top 10. CF는 최대 100명 이웃과 미평가 책 20권, 직접 취향과 50:50 결합이다. CF가 불가능하면 직접 취향을 그대로 사용한다. 원래 비교의 5만 사전 후보 범위는 현재 책 텍스트에서 다시 계산한다. 영화 평점은 평가만 담당한다.

## 검증 범위

작은 원본 형태 데이터와 실제 MiniLM 모델로 네 모델의 입력부터 추천·평가까지 검증했다. 기존 CF와 대조한 표본에서 선택된 책과 가중치가 일치했다. 전체 데이터로 새 코드와 기존 결과의 완전한 순위 동등성은 아직 검증하지 못했다. 부동소수점 계산 순서가 달라 경계 순위가 달라질 수 있다. 보고서 수치는 reference_metrics.json이며 전체 재실행 결과와 구분해야 한다.

## 폴더 정리

과거 실험과 임시 파일은 보조자료/에 모았다. 실행에는 data/, models/, src/, main.py, stopwords.json이 필요하다. 제출 ZIP에는 코드와 설치 안내, 불용어 설정, 기존 평가표를 담았으며 원본 데이터와 모델 파일은 별도로 준비한다.
