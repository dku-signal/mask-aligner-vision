# YOLO + SAM2 모델 실험 가이드

[프로젝트 소개](../README.md) · [장비 실행](team-guide.md) · [검증 범위](validation.md)

## 모델을 어디에 쓰는가

`sam2_keys.py`의 `YoloSam2Keys`는 **YOLO 사각형 박스 → SAM2 분할 → 사각형 기하 검사 → PreciseKeys 측정**을 수행합니다. 사용 모델은 `facebook/sam2.1-hiera-small`입니다. 웨이퍼 십자 탐색은 현재 저장소의 기존 경로를 사용합니다.

별도 SW 실험에서는 YOLO로 마스크 사각형 4개와 웨이퍼 키 1개를 찾고 다섯 박스를 SAM2에 전달했습니다. README의 2026-10-02 관찰 결과는 이 별도 경로에서 나온 것입니다. 현재 저장소가 두 구현을 모두 포함한다고 가정하지 않습니다.

## 설치와 가중치

기본 가상환경에서:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
python -m pip install ultralytics
```

현재 `requirements-optional.txt`에는 Ultralytics가 없어 별도 설치가 필요합니다. 이 문서 수정 작업에서는 패키지 설치나 모델 재학습을 실행하지 않았습니다. 실제 사용 환경의 torch/SAM2 버전은 실행 결과와 함께 기록합니다.

가중치는 다음 위치에 준비합니다.

```text
runs/keys/weights/best.pt
```

`sam2_keys.py`의 `YOLO_WEIGHTS`가 이 경로를 참조합니다. 현재 앱 CLI에는 별도 YOLO `--weights` 옵션이 없습니다. **이 표식을 학습한 가중치**를 사용해야 합니다. 가중치는 `.gitignore` 대상으로 저장소에 포함하지 않습니다.

현재 구현은 YOLO 클래스 ID **0**을 마스크 사각형으로 사용합니다. 저장소의 `tools/make_dataset.py`가 만드는 합성 데이터는 `0: square`, `1: cross`입니다. 별도 실제 영상 실험의 이름은 `0: mask_square`, `1: wafer_key`였습니다. 클래스 이름·순서가 다른 가중치를 무조건 복사해서 사용하면 안 됩니다.

## 실제 데이터 학습 설계

1. 마스크 사각형 하나마다 박스를 붙입니다. 정상적인 한 묶음에는 네 사각형이 있습니다.
2. 웨이퍼 십자 전체에 별도 클래스 박스를 붙입니다.
3. 위치·회전·조명·배율·겹침이 다양한 실제 영상을 수집합니다.
4. 촬영 세션 단위로 train/validation/test를 나눕니다. 유사 연속 프레임을 양쪽에 섞지 않습니다.
5. 검출 mAP 외에 두 키 구분 성공률, 중심·각도 오차, 잘못된 성공 판정, 지연을 함께 측정합니다.

초기 YOLO26n 실험의 9장 데이터는 파이프라인 연결을 확인하기 위한 자료입니다. 높은 학습 지표를 일반화 성능으로 소개하지 않습니다. [학습 manifest](evidence/yolo-trial-manifest.json)

SAM2는 현재 사전 학습 모델을 그대로 사용합니다. YOLO 학습을 위한 박스 라벨과 SAM2 재학습용 픽셀 마스크 라벨은 서로 다른 자료입니다.

## 실행 옵션

```bash
python mask_aligner_optimized_v4.py --camera auto --detector yolo-sam2
```

이 명령은 장비 제어 웹 앱을 엽니다. 모델 준비 후에도 장비 실행 절차와 교정 조건은 [팀 가이드](team-guide.md)를 따릅니다.

같은 **저장 이미지**에서 검출기만 비교하려면:

```bash
python tools/test_detectors.py path/to/key-image.png   --detectors opencv,yolo-sam2 --runs 3
```

위 저장 이미지 비교는 모터를 구동하지 않습니다. `--camera 0`은 새 카메라 프레임을 입력으로 쓰는 옵션입니다. 합성 이미지 도구는 `tools/make_dataset.py`, 측정 정밀도 벤치마크는 `tools/bench_precision.py`에 있습니다.

| 옵션 | 후보 영역 생성 방식 |
|---|---|
| `opencv` | 규칙 기반 후보 + 서브픽셀 측정, 현재 기본값 |
| `opencv-v4` | 기존 정수 픽셀 측정 비교 |
| `sam2` | SAM2 자동 분할 후보 |
| `hybrid` | SAM2와 OpenCV 사각형 후보 결합 |
| `yolo-sam2` | YOLO 사각형 박스 → SAM2 |
| `opencv-sam2` | OpenCV 사각형 박스 → SAM2 |

## 이동 후 평가는 무엇인가

**팀 앱**은 새 영상의 동일 검출 경로로 오차를 재측정하여 다음 이동 또는 완료를 결정합니다. 별도 평가 모델을 학습한 구조가 아닙니다.

**별도 SW의 단일 계획 실험**은 이동 후 새 프레임에서 `YOLO → 원본 경계 측정`을 수행합니다. SAM2를 다시 실행하지 않고, 평가기에는 모터·플래너 참조가 없어 추가 이동으로 돌아가지 않습니다. 관찰 전용 실행에서는 실제 이동 없이 서로 다른 전후 프레임을 사용해 이 측정 흐름을 확인했습니다.

## 결과 해석

- YOLO confidence: 후보의 검출 점수.
- SAM2 score: 분할 품질에 대한 모델의 추정값.
- 기하 잔차: 관측 경계가 피팅한 도형을 얼마나 잘 따르는지 나타내는 값.
- dx/dy/영상 각도: 두 기준 사이의 영상상 차이.
- μm: 물리 길이 교정 및 그 불확도까지 확인한 뒤에만 성능 지표로 사용.

검출 성공이 최종 정렬 성공을 의미하지 않습니다. 특히 가림이나 θ 오차가 있을 때 중심 정렬과 네 모서리 전체 일치를 별도로 확인합니다.
