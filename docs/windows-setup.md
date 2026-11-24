# 윈도우 노트북 세팅 (통합 스테이션)

[프로젝트 소개](../README.md) · [통합 스테이션 사용법](station.md) · [장비 인계 자료](handover-2026-11-20.md)

새 윈도우 노트북에서 GitHub 저장소를 받아 `tools/station.py`(프리얼라인 → 정밀 얼라인 연속 운전)를 돌리기까지 필요한 것을 순서대로 적었습니다. 명령은 모두 **PowerShell** 기준입니다.

> macOS에서 확인한 절차를 윈도우 명령으로 옮긴 것이며, 윈도우 실기에서 처음 세팅할 때 막힌 곳이 있으면 이 문서에 추가해 주세요.

## 1. 설치할 프로그램

| 프로그램 | 용도 | 비고 |
|---|---|---|
| [Python 3.11](https://www.python.org/downloads/windows/) | 앱 실행 | 설치 화면에서 **Add python.exe to PATH** 체크. 3.12 이상은 확인하지 않음 |
| [Git for Windows](https://git-scm.com/download/win) | 저장소 받기·업데이트 | 기본 옵션 그대로 |
| [Arduino IDE 2](https://www.arduino.cc/en/software) | 펌웨어 v2 업로드 | Mega 드라이버가 같이 설치됨 |

저장소와 가중치는 공개라 GitHub 로그인 없이 받을 수 있습니다. 코드를 올리는(push) 사람만 로그인이 필요합니다.

## 2. 코드 받기

OneDrive로 동기화되는 폴더(바탕화면·문서)는 피하세요. 실행 중 `out/`에 카메라 프레임이 저장되는데, 사람 얼굴이 찍힐 수 있어 클라우드로 올라가면 안 됩니다. 한글이 들어간 경로도 피하는 편이 안전합니다.

```powershell
mkdir C:\work -Force
cd C:\work
git clone https://github.com/dku-signal/mask-aligner-vision.git
cd mask-aligner-vision
```

## 3. 파이썬 환경

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ultralytics pytest
```

- `Activate.ps1`이 "스크립트를 실행할 수 없습니다"로 막히면 한 번만 실행: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- `ultralytics`가 torch(CPU용)를 같이 설치합니다. NVIDIA 그래픽카드가 있으면 [pytorch.org](https://pytorch.org/get-started/locally/)에서 CUDA용 torch를 **먼저** 설치한 뒤 `ultralytics`를 설치하면 YOLO가 GPU로 돕니다(`two_pair.py`가 자동 선택).
- SAM2(`requirements-optional.txt`)는 필요 없습니다. 현재 정밀 얼라인은 `yolo-2pair`(YOLO + OpenCV)입니다.

macOS에서 확인한 버전(참고용):

| 패키지 | 버전 |
|---|---|
| Python | 3.11 |
| opencv-python | 5.0.0 |
| numpy | 2.4.6 |
| pyserial | 3.5 |
| torch | 2.14.0 |
| ultralytics | 8.4.164 |

이후 새 PowerShell 창을 열 때마다 `cd C:\work\mask-aligner-vision` → `.venv\Scripts\Activate.ps1` 후 실행합니다.

## 4. YOLO 가중치

가중치는 저장소에 없고 [Release `weights-v1`](https://github.com/dku-signal/mask-aligner-vision/releases/tag/weights-v1)에 있습니다. 이 경로에 두면 `--detector yolo-2pair`와 프리얼라인 측정이 자동으로 씁니다.

```powershell
mkdir runs\keys_real\weights -Force
Invoke-WebRequest https://github.com/dku-signal/mask-aligner-vision/releases/download/weights-v1/keys_real_best.pt -OutFile runs\keys_real\weights\best.pt
(Get-FileHash runs\keys_real\weights\best.pt -Algorithm SHA256).Hash
```

마지막 줄 출력이 `42DCA8F5734DDB1A85F4D1A2B0CAC30AAB075199B17A2D5422DAD9C619D460A6`이면 정상입니다.

## 5. 장비 없이 설치 확인

```powershell
python -m pytest -q
```

`62 passed`가 나오면 코드와 패키지는 준비된 것입니다.

```powershell
python tools\station.py --mock --cameras --fine-camera 9
```

가상 보드로 통합 화면을 띄웁니다. 브라우저에서 `http://127.0.0.1:8790`(통합 화면)과 `http://127.0.0.1:8765`(정밀 화면)가 열리면 됩니다. `--cameras`(빈 값)와 `--fine-camera 9`(없는 번호)는 노트북 내장 웹캠이 켜지지 않게 하려는 것이라, 화면에 "카메라 열기 실패"가 보이는 것은 정상입니다. `Ctrl+C`로 끕니다.

## 6. 펌웨어 v2 업로드 (보드에 한 번)

통합 스테이션은 **펌웨어 v2**(`firmware/mega_previous`, 2026-11-24)가 있어야 정밀 얼라인과 이송을 동시에 움직일 수 있습니다. 보드에 v1이 올라가 있으면 먼저 올립니다.

1. **SMPS(모터 전원) OFF.** USB만 연결합니다.
2. Arduino IDE → **파일 → 열기** → `C:\work\mask-aligner-vision\firmware\mega_previous\mega_previous.ino`
3. **도구 → 보드 → Arduino Mega or Mega 2560**, **도구 → 포트 → COM 번호** (장치 관리자 "포트(COM & LPT)"에서 Arduino Mega로 보이는 번호)
4. **라이브러리 매니저**에서 `Adafruit ADS1X15` 설치 (Servo는 기본 포함)
5. **업로드**. 끝나면 **시리얼 모니터를 닫고** IDE도 닫습니다(포트를 잡고 있으면 앱이 연결하지 못함).

주의:
- 하드웨어 팀의 원본 스케치를 그대로 올리지 마세요. 연속회전 서보가 부팅하자마자 계속 돕니다(v2는 이를 고친 버전).
- `firmware/aligner_mega`는 11/5 배선용이라 현재 장비와 핀이 다릅니다. 올리지 마세요.

## 7. 장비 연결과 첫 실행

1. SMPS OFF 상태에서 아두이노 USB와 카메라 2대(0: 정밀·마스크, 1: 프리얼라인 웨이퍼)를 연결합니다.
2. 실행:
   ```powershell
   python tools\station.py --detector yolo-2pair
   ```
3. 통합 화면(`http://127.0.0.1:8790`)에서 보드 "연결됨"과 카메라 영상을 확인합니다.
   - **카메라 번호가 맥과 다를 수 있습니다.** 노트북 내장 웹캠이 보통 0번을 차지해 장비 카메라가 1·2번이 됩니다. 화면을 보고 역할에 맞게 다시 실행하세요. 예: `python tools\station.py --detector yolo-2pair --fine-camera 1 --cameras 1 2`
   - 포트를 못 찾으면 직접 지정: `--port COM5`
   - YOLO가 CPU에서 느리면 정밀 카메라 해상도를 낮춥니다: `--frame 1280x720`
4. 모두 확인되면 **SMPS ON**.
5. **이 노트북에서 처음 한 번:** 정밀 X·Y를 장비의 오른쪽 위 끝에 두고 정밀 화면의 **현재 위치를 오른쪽 위 기준으로 저장**을 누릅니다. 이 기준과 축 위치 기록은 `out\`에 저장되며 git으로 옮겨지지 않으므로, 다른 컴퓨터에서 쓰던 값은 이어지지 않습니다.
6. 이후 사용법(연속 운전 시작, 스테이지 비움 확인, 끝나는 조건)은 [station.md](station.md)를 따릅니다. 처음에는 **최대 장수 1**로 돌려 보세요.

알아둘 점:
- 앱이 시리얼 포트를 열 때마다 Mega가 리셋되어 **전자석이 꺼집니다.** 웨이퍼를 전자석으로 들고 있는 동안에는 앱을 끄거나 다시 켜지 마세요.
- `tools\motor_panel.py`와 `mask_aligner_optimized_v4.py`를 따로 실행할 수도 있지만, 보드는 한 번에 한 프로그램만 잡습니다. `station.py`와 동시에 켜지 마세요.

## 8. 자주 막히는 곳

| 증상 | 원인 · 해결 |
|---|---|
| `python`을 치면 Microsoft Store가 열림 | 설정 → 앱 → 고급 앱 설정 → **앱 실행 별칭**에서 python 두 개를 끄거나, 가상환경을 활성화한 뒤 실행 |
| `Activate.ps1` 실행 거부 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 포트 연결 실패 `PermissionError` / 액세스 거부 | Arduino IDE 시리얼 모니터나 다른 앱이 COM 포트를 잡고 있음. 모두 닫고 다시 실행 |
| 보드가 목록에 안 보임 | USB 케이블(충전 전용 아닌지), 장치 관리자에서 COM 번호 확인. 정품이 아닌 CH340 보드면 CH340 드라이버 설치 |
| 카메라가 안 열림 / 검은 화면 | 설정 → 개인 정보 → **카메라 → 데스크톱 앱의 카메라 액세스 허용** 켜기. Teams·Zoom·카메라 앱이 카메라를 잡고 있으면 끄기 |
| 카메라 역할이 바뀜 | USB 꽂는 순서에 따라 번호가 바뀜. 화면을 보고 `--fine-camera`, `--cameras`로 지정 |
| 방화벽 허용 창 | 앱은 `127.0.0.1`에서만 열려 외부 접속이 필요 없음. 허용하지 않아도 됨 |
| 정밀 얼라인 인식이 느림 | `--frame 1280x720`, NVIDIA GPU면 CUDA용 torch 설치(3단계) |

## 9. 업데이트

```powershell
cd C:\work\mask-aligner-vision
git pull
python -m pip install -r requirements.txt
```

펌웨어(`firmware/mega_previous`)가 바뀐 커밋을 받았다면 6단계로 다시 업로드합니다.

**올리면 안 되는 것:** `out\`, `data\`(카메라 프레임, 얼굴 포함 가능)와 `runs\`(가중치)는 `.gitignore`에 들어 있습니다. `git add -f`로 억지로 넣지 마세요.
