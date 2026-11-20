#include <Wire.h>
#include <Servo.h>
#include <Adafruit_ADS1X15.h>
#include <avr/interrupt.h>
#include <util/atomic.h>

#if !defined(__AVR_ATmega2560__)
#error "Select Arduino Mega 2560."
#endif

// MEGA_PREVIOUS (patched) - Arduino Mega 2560 / Serial 9600 baud
// F/B: X STEP D48, DIR D49
// f/b: Y STEP D22, DIR D23
// Q/W: rail 1 STEP D52, DIR D53
// J/H: rail 1 + rail 2 together (same pulses)
// E/R: rail 2 STEP D26, DIR D27
// u/d: original Z IN1 D2, IN2 D3
// O/P: extra Z 1 IN1 D34, IN2 D35
// K/L: extra Z 2 IN1 D38, IN2 D39
// T/G: original theta 90/0 degrees, D5
// t: extra theta 0/90 toggle, D8
// M: magnet ON/OFF, D11 (active HIGH driver input)
// X/x: stop all rails and Z axes; retain magnet and servo states.
// Line commands (start with '#', end with newline) finish by themselves and reply:
//   #<F|B|f|b|Q|W|E|R|J|H> <steps>   finite rail move -> "DONE <c> <steps>"
//   #<u|d|O|P|K|L> <ms>              timed DC move    -> "DONE <c> <ms>"
//   #t <ms> <+1|-1>                  pre-align continuous servo timed turn -> "DONE t <ms>"
//   #A <ms> <+1|-1>                  precision theta continuous servo (D5) timed turn -> "DONE A <ms>"
//   #M <0|1>                         magnet off/on (absolute) -> "DONE M <0|1>"
//   #PING                            -> "PONG"
// X/x also cancels a pending line command and replies "STOP". If a rail or servo was stopped
// part-way it first reports how far it actually went: "STOP <c> <steps|ms>" (for the PC ledger).
// Four STEP/DIR rails share one pulse generator: one rail moves at a time.
// Z axes operate independently. No limit switches or automatic travel stop.
// ADS1115 laser measurement and calibration retained from supplied sketch.
// Timer2 reserved for STEP generation; do not use tone()/Timer2 PWM.
// Motors and magnet require suitable external drivers.

// ------------------------
// 레일 핀
// ------------------------
const byte X_STEP_PIN = 48;
const byte X_DIR_PIN  = 49;

const byte Y_STEP_PIN = 22;
const byte Y_DIR_PIN  = 23;

const byte RAIL1_STEP_PIN = 52;
const byte RAIL1_DIR_PIN  = 53;

const byte RAIL2_STEP_PIN = 26;
const byte RAIL2_DIR_PIN  = 27;

// ------------------------
// Z축 핀
// ------------------------
const byte Z1_IN1_PIN = 2;
const byte Z1_IN2_PIN = 3;

const byte Z2_IN1_PIN = 34;
const byte Z2_IN2_PIN = 35;
const byte Z3_IN1_PIN = 38;
const byte Z3_IN2_PIN = 39;

// ------------------------
// 세타 핀 및 각도
// ------------------------
const byte THETA_PIN = 5;
const byte PREALIGN_PIN = 8;

const int THETA_ANGLE_A = 0;
const int THETA_ANGLE_B = 90;

const int PREALIGN_ANGLE_A = 0;
const int PREALIGN_ANGLE_B = 90;

const int SERVO_STOP = 90;

Servo thetaServo;
Servo prealignServo;

bool thetaAtB = false;
bool prealignAtB = false;

// ------------------------
// 전자석 구동회로 제어 핀
// HIGH=ON인 구동회로 기준
// ------------------------
const byte MAGNET1_PIN = 11;

bool magnet1On = false;

// ------------------------
// 레일 STEP 상태
// Timer2: 50us마다 신호 반전
// 전체 펄스 주기: 약 100us
// ------------------------
const byte NO_ACTIVE_RAIL = 255;

volatile byte activeStepPin = NO_ACTIVE_RAIL;
volatile byte activeStepPin2 = NO_ACTIVE_RAIL;  // 이송 레일 1+2 동시 구동용

// 이송부 양쪽 모터가 서로 반대로 설치돼 있으면 true로 바꾼다
const bool RAIL2_PAIR_INVERT = false;
volatile bool stepState = HIGH;
volatile long stepsLeft = -1;          // -1 = 수동(무제한), >0 = 남은 스텝
volatile bool railDone = false;
char railCmd = 0;
long railTotal = 0;

char dcCmd = 0;
unsigned long dcStart = 0, dcMs = 0;
bool servoTurning = false;
unsigned long servoStart = 0, servoMs = 0;
bool thetaTurning = false;
unsigned long thetaStart = 0, thetaMs = 0;
const int THETA_TURN_SPEED = 30;       // 정밀 θ 연속회전 서보: 90±30

const int PREALIGN_TURN_SPEED = 30;    // 연속회전 서보: 90±30 으로 회전
char lineBuf[32];
byte lineLen = 0;
bool inLine = false;

// ------------------------
// ADS1115 및 센서
//
// VDD  -> 외부 안정화 5V
// GND  -> 외부 0V 및 Mega GND 공통
// SDA  -> Mega D20
// SCL  -> Mega D21
// ADDR -> GND
//
// 센서 회색선 -- 30kΩ --+-- ADS1115 A0
//                      |
//                     30kΩ
//                      |
//                     GND
//
// 센서 실드선 AGND -> ADS1115 GND
// A0-GND 사이에 0.1uF
//
// 센서: 전압 출력 / diSP=Std
// Zero set 및 Peak/Bottom hold 해제
// Hold OFF
// ------------------------
Adafruit_ADS1115 ads;

const uint8_t ADS_ADDRESS = 0x48;
const float DIVIDER_FACTOR = 2.0f;
const float ALARM_THRESHOLD_V = 5.10f;

// 최신 3점 보정: 이전 출력 -> 센서 표시
// -0.60 -> -0.81
// -11.95 -> -12.02
// 12.88 -> 12.54
const float CAL_SLOPE = -0.995459184f;
// 센서 표시 대비 -0.02mm 차이를 보완: 기존 오프셋 +0.02mm
const float CAL_OFFSET = 49.671333329f;

const unsigned long SENSOR_INTERVAL_MS = 200;
const unsigned long SENSOR_POLL_MS = 2;
const unsigned long SENSOR_TIMEOUT_MS = 100;

bool adsReady = false;
bool adcPending = false;

unsigned long lastSensorStart = 0;
unsigned long adcStartTime = 0;
unsigned long lastAdcPoll = 0;
unsigned long lastAdsError = 0;

// ------------------------
// 함수 선언
// ------------------------
void initRail(byte stepPin, byte dirPin);
void initZ(byte in1Pin, byte in2Pin);
void initStepTimer();
void stopRails();
void startRail(byte stepPin, byte dirPin, byte direction);
void stopZ(byte in1Pin, byte in2Pin);
void moveZ(byte in1Pin, byte in2Pin, bool upward);
void stopAllMotion();
void handleCommand(char command);
void updateSensor();
void printSensorValue(int16_t raw);
bool takeWireTimeout();
void clearWireTimeout();

// ========================================
// Timer2 STEP生成
// ADCやシリアル処理とは独立して実行
// ========================================
ISR(TIMER2_COMPA_vect) {
  byte pin = activeStepPin;

  if (pin != NO_ACTIVE_RAIL) {
    stepState = !stepState;
    digitalWrite(pin, stepState ? HIGH : LOW);

    byte pin2 = activeStepPin2;
    if (pin2 != NO_ACTIVE_RAIL) {
      digitalWrite(pin2, stepState ? HIGH : LOW);
    }

    // 한 펄스가 끝나면(HIGH 복귀) 남은 스텝 감소, 0이면 스스로 멈춘다
    if (stepState && stepsLeft > 0 && --stepsLeft == 0) {
      TIMSK2 &= ~_BV(OCIE2A);
      activeStepPin = NO_ACTIVE_RAIL;
      activeStepPin2 = NO_ACTIVE_RAIL;
      railDone = true;
    }
  }
}

void setup() {
  Serial.begin(9600);

  // 모든 구동 출력 초기화
  initRail(X_STEP_PIN, X_DIR_PIN);
  initRail(Y_STEP_PIN, Y_DIR_PIN);
  initRail(RAIL1_STEP_PIN, RAIL1_DIR_PIN);
  initRail(RAIL2_STEP_PIN, RAIL2_DIR_PIN);

  initZ(Z1_IN1_PIN, Z1_IN2_PIN);
  initZ(Z2_IN1_PIN, Z2_IN2_PIN);
  initZ(Z3_IN1_PIN, Z3_IN2_PIN);

  // 전자석 초기 OFF
  digitalWrite(MAGNET1_PIN, LOW);
  pinMode(MAGNET1_PIN, OUTPUT);

  initStepTimer();

  // 연속회전 서보는 90이 정지 신호 → 전원 인가 시 정지 상태로 시작
  thetaServo.write(SERVO_STOP);
  thetaServo.attach(THETA_PIN);

  prealignServo.write(SERVO_STOP);
  prealignServo.attach(PREALIGN_PIN);

  // I2C 초기화
  Wire.begin();
  // 설치된 AVR 코어(1.8.8)는 WIRE_HAS_TIMEOUT 매크로가 없어 아래 #if가 꺼진다.
  // 타임아웃이 없으면 ADS1115 전원/배선 문제 시 setup이 영원히 멈추므로 직접 켠다.
  Wire.setWireTimeout(25000, true);

#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(25000, true);
#endif

  // ADS1115 전원이 없으면 ads.begin()에서 멈추므로 주소 응답부터 확인
  Wire.beginTransmission(ADS_ADDRESS);
  bool adsAck = (Wire.endTransmission() == 0);
  clearWireTimeout();
  adsReady = adsAck && ads.begin(ADS_ADDRESS, &Wire);

  if (adsReady) {
    ads.setGain(GAIN_ONE);
    ads.setDataRate(RATE_ADS1115_128SPS);
  }

  clearWireTimeout();

  // 시작 안내 문구 없이 측정값만 출력
  lastSensorStart = millis();
  lastAdsError = millis();
  Serial.println(F("READY"));
}

bool isRailCmd(char c) { return strchr("FBfbQWERJH", c) != NULL; }
bool isDcCmd(char c) { return strchr("udOPKL", c) != NULL; }

void runLine(char *line) {
  char c = line[0];
  long n = 0, dir = 0;
  if (strcmp(line, "PING") == 0) { Serial.println(F("PONG")); return; }
  if (c == 'M' && sscanf(line + 1, "%ld", &n) == 1) {
    magnet1On = n != 0;
    digitalWrite(MAGNET1_PIN, magnet1On ? HIGH : LOW);
    Serial.print(F("DONE M ")); Serial.println(magnet1On ? 1 : 0);
    return;
  }
  if (c == 't' && sscanf(line + 1, "%ld %ld", &n, &dir) == 2 && n > 0 && n <= 20000) {
    prealignServo.write(SERVO_STOP + (dir >= 0 ? PREALIGN_TURN_SPEED : -PREALIGN_TURN_SPEED));
    servoTurning = true; servoStart = millis(); servoMs = n;
    return;
  }
  if (c == 'A' && sscanf(line + 1, "%ld %ld", &n, &dir) == 2 && n > 0 && n <= 20000) {
    thetaServo.write(SERVO_STOP + (dir >= 0 ? THETA_TURN_SPEED : -THETA_TURN_SPEED));
    thetaTurning = true; thetaStart = millis(); thetaMs = n;
    return;
  }
  if (sscanf(line + 1, "%ld", &n) != 1 || n <= 0) { Serial.print(F("ERR ")); Serial.println(line); return; }
  if (isRailCmd(c)) {
    if (n > 2000000L) { Serial.println(F("ERR steps")); return; }
    railCmd = c; railTotal = n; railDone = false;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { stepsLeft = n; }
    handleCommand(c);
  } else if (isDcCmd(c)) {
    if (n > 30000) { Serial.println(F("ERR ms")); return; }
    dcCmd = c; dcStart = millis(); dcMs = n;
    handleCommand(c);
  } else {
    Serial.print(F("ERR ")); Serial.println(line);
  }
}

void serviceTimers() {
  if (railDone) {
    railDone = false;
    stepsLeft = -1;
    stepState = HIGH;
    Serial.print(F("DONE ")); Serial.print(railCmd); Serial.print(' '); Serial.println(railTotal);
    railCmd = 0;
  }
  if (dcCmd && millis() - dcStart >= dcMs) {
    stopZ(Z1_IN1_PIN, Z1_IN2_PIN); stopZ(Z2_IN1_PIN, Z2_IN2_PIN); stopZ(Z3_IN1_PIN, Z3_IN2_PIN);
    Serial.print(F("DONE ")); Serial.print(dcCmd); Serial.print(' '); Serial.println(dcMs);
    dcCmd = 0;
  }
  if (servoTurning && millis() - servoStart >= servoMs) {
    prealignServo.write(SERVO_STOP);
    servoTurning = false;
    Serial.print(F("DONE t ")); Serial.println(servoMs);
  }
  if (thetaTurning && millis() - thetaStart >= thetaMs) {
    thetaServo.write(SERVO_STOP);
    thetaTurning = false;
    Serial.print(F("DONE A ")); Serial.println(thetaMs);
  }
}

void loop() {
  // 사용자 명령 처리: '#'로 시작하면 줄 명령, 아니면 한 글자 수동 명령
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (inLine) {
      if (ch == '\n' || ch == '\r') {
        lineBuf[lineLen] = 0; inLine = false;
        if (lineLen) runLine(lineBuf);
      } else if (lineLen < sizeof(lineBuf) - 1) {
        lineBuf[lineLen++] = ch;
      }
    } else if (ch == '#') {
      inLine = true; lineLen = 0;
    } else {
      handleCommand(ch);
    }
  }
  serviceTimers();


  // 변환 대기 루프 없이 센서 처리
  updateSensor();
}

// ========================================
// 출력 초기화
// ========================================
void initRail(byte stepPin, byte dirPin) {
  // Common-Anode STEP 대기 상태 HIGH
  digitalWrite(stepPin, HIGH);
  digitalWrite(dirPin, LOW);

  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
}

void initZ(byte in1Pin, byte in2Pin) {
  digitalWrite(in1Pin, LOW);
  digitalWrite(in2Pin, LOW);

  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);
}

// ========================================
// Timer2 설정
// Mega 16MHz / 8 = 2MHz
// 100 카운트 = 50us
// ========================================
void initStepTimer() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    TCCR2A = 0;
    TCCR2B = 0;
    TIMSK2 = 0;
    TCNT2 = 0;

    OCR2A = 99;

    // CTC 모드
    TCCR2A = _BV(WGM21);

    // 분주비 8
    TCCR2B = _BV(CS21);

    // 기존 비교 일치 플래그 제거
    TIFR2 = _BV(OCF2A);
  }
}

// ========================================
// 레일 제어
// ========================================
void stopRails() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    TIMSK2 &= ~_BV(OCIE2A);

    activeStepPin = NO_ACTIVE_RAIL;
    activeStepPin2 = NO_ACTIVE_RAIL;
    stepState = HIGH;

    digitalWrite(X_STEP_PIN, HIGH);
    digitalWrite(Y_STEP_PIN, HIGH);
    digitalWrite(RAIL1_STEP_PIN, HIGH);
    digitalWrite(RAIL2_STEP_PIN, HIGH);
  }
}

void startRail(byte stepPin, byte dirPin, byte direction) {
  stopRails();

  digitalWrite(dirPin, direction);
  delayMicroseconds(100);

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    stepState = HIGH;
    activeStepPin = stepPin;

    TCNT2 = 0;
    TIFR2 = _BV(OCF2A);
    TIMSK2 |= _BV(OCIE2A);
  }
}

// ========================================
// Z축 제어
// LOW/LOW=정지인 드라이버 기준
// ========================================
// 이송 레일 1과 2를 같은 펄스로 동시에 구동
void startRailPair(byte direction) {
  stopRails();

  digitalWrite(RAIL1_DIR_PIN, direction);
  digitalWrite(RAIL2_DIR_PIN, RAIL2_PAIR_INVERT ? !direction : direction);
  delayMicroseconds(100);

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    stepState = HIGH;
    activeStepPin = RAIL1_STEP_PIN;
    activeStepPin2 = RAIL2_STEP_PIN;

    TCNT2 = 0;
    TIFR2 = _BV(OCF2A);
    TIMSK2 |= _BV(OCIE2A);
  }
}

void stopZ(byte in1Pin, byte in2Pin) {
  digitalWrite(in1Pin, LOW);
  digitalWrite(in2Pin, LOW);
}

void moveZ(byte in1Pin, byte in2Pin, bool upward) {
  stopZ(in1Pin, in2Pin);

  if (upward) {
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
  } else {
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, HIGH);
  }
}

void stopAllMotion() {
  bool pending = railCmd || dcCmd || servoTurning || thetaTurning;
  // 중간 정지 시 실제로 간 양을 알려 PC 장부가 어긋나지 않게 한다: "STOP <c> <n>"
  long railDoneSteps = 0;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    if (railCmd && stepsLeft > 0) railDoneSteps = railTotal - stepsLeft;
  }
  char stoppedRail = railCmd;
  unsigned long servoDoneMs = servoTurning ? millis() - servoStart : 0;
  unsigned long thetaDoneMs = thetaTurning ? millis() - thetaStart : 0;
  if (thetaTurning) { thetaServo.write(SERVO_STOP); thetaTurning = false; }
  stopRails();
  stepsLeft = -1; railDone = false; railCmd = 0; dcCmd = 0;
  if (servoTurning) { prealignServo.write(SERVO_STOP); servoTurning = false; }
  if (stoppedRail && railDoneSteps > 0) { Serial.print(F("STOP ")); Serial.print(stoppedRail); Serial.print(' '); Serial.println(railDoneSteps); }
  if (servoDoneMs) { Serial.print(F("STOP t ")); Serial.println(servoDoneMs); }
  if (thetaDoneMs) { Serial.print(F("STOP A ")); Serial.println(thetaDoneMs); }
  if (pending) Serial.println(F("STOP"));

  stopZ(Z1_IN1_PIN, Z1_IN2_PIN);
  stopZ(Z2_IN1_PIN, Z2_IN2_PIN);
  stopZ(Z3_IN1_PIN, Z3_IN2_PIN);
}

// ========================================
// 시리얼 명령
// 정상 명령 응답 문구는 출력하지 않음
// ========================================
void handleCommand(char command) {
  switch (command) {
    case 'F':
      startRail(X_STEP_PIN, X_DIR_PIN, LOW);
      break;

    case 'B':
      startRail(X_STEP_PIN, X_DIR_PIN, HIGH);
      break;

    case 'f':
      startRail(Y_STEP_PIN, Y_DIR_PIN, LOW);
      break;

    case 'b':
      startRail(Y_STEP_PIN, Y_DIR_PIN, HIGH);
      break;

    case 'Q':
      startRail(RAIL1_STEP_PIN, RAIL1_DIR_PIN, LOW);
      break;

    case 'W':
      startRail(RAIL1_STEP_PIN, RAIL1_DIR_PIN, HIGH);
      break;

    case 'E':
      startRail(RAIL2_STEP_PIN, RAIL2_DIR_PIN, LOW);
      break;

    case 'R':
      startRail(RAIL2_STEP_PIN, RAIL2_DIR_PIN, HIGH);
      break;

    case 'J':
      startRailPair(LOW);
      break;

    case 'H':
      startRailPair(HIGH);
      break;

    case 'u':
      moveZ(Z1_IN1_PIN, Z1_IN2_PIN, true);
      break;

    case 'd':
      moveZ(Z1_IN1_PIN, Z1_IN2_PIN, false);
      break;

    case 'O':
      moveZ(Z2_IN1_PIN, Z2_IN2_PIN, true);
      break;

    case 'P':
      moveZ(Z2_IN1_PIN, Z2_IN2_PIN, false);
      break;

    case 'K':
      moveZ(Z3_IN1_PIN, Z3_IN2_PIN, true);
      break;

    case 'L':
      moveZ(Z3_IN1_PIN, Z3_IN2_PIN, false);
      break;

    case 'T':
      thetaAtB = true;
      thetaServo.write(THETA_ANGLE_B);
      break;

    case 'G':
      thetaAtB = false;
      thetaServo.write(THETA_ANGLE_A);
      break;

    case 't':
      prealignAtB = !prealignAtB;
      prealignServo.write(
          prealignAtB ? PREALIGN_ANGLE_B : PREALIGN_ANGLE_A
      );
      break;

    case 'M':
      magnet1On = !magnet1On;
      digitalWrite(MAGNET1_PIN, magnet1On ? HIGH : LOW);
      break;

    case 'X':
    case 'x':
      stopAllMotion();
      break;

    default:
      // 줄바꿈 및 기타 문자 무시
      break;
  }
}

// ========================================
// I2C 타임아웃 처리
// ========================================
void clearWireTimeout() {
  Wire.clearWireTimeoutFlag();
}

bool takeWireTimeout() {
  bool timedOut = Wire.getWireTimeoutFlag();
  Wire.clearWireTimeoutFlag();
  return timedOut;
}

// ========================================
// 센서 비동기 변환 처리
// 시작 → 다른 작업 → 완료 확인 → 읽기
// ========================================
void updateSensor() {
  unsigned long now = millis();

  if (!adsReady) {
    if (now - lastAdsError >= 1000) {
      lastAdsError = now;
      Serial.println(F("ADS_ERROR"));
    }
    return;
  }

  if (!adcPending) {
    if (now - lastSensorStart < SENSOR_INTERVAL_MS) {
      return;
    }

    lastSensorStart = now;
    clearWireTimeout();

    // A0를 GND 기준으로 단발 변환
    ads.startADCReading(
        ADS1X15_REG_CONFIG_MUX_SINGLE_0,
        false
    );

    if (takeWireTimeout()) {
      Serial.println(F("ADS_ERROR"));
      return;
    }

    adcStartTime = millis();
    lastAdcPoll = adcStartTime;
    adcPending = true;
    return;
  }

  if (now - adcStartTime >= SENSOR_TIMEOUT_MS) {
    adcPending = false;
    Serial.println(F("ADS_ERROR"));
    return;
  }

  if (now - lastAdcPoll < SENSOR_POLL_MS) {
    return;
  }

  lastAdcPoll = now;

  bool complete = ads.conversionComplete();

  if (takeWireTimeout()) {
    adcPending = false;
    Serial.println(F("ADS_ERROR"));
    return;
  }

  if (!complete) {
    return;
  }

  int16_t raw = ads.getLastConversionResults();
  adcPending = false;

  if (takeWireTimeout()) {
    Serial.println(F("ADS_ERROR"));
    return;
  }

  printSensorValue(raw);
}

// ========================================
// 보정된 센서 표시값 출력
// ========================================
void printSensorValue(int16_t raw) {
  float adcVoltage = ads.computeVolts(raw);
  float sensorVoltage = adcVoltage * DIVIDER_FACTOR;

  if (raw == 32767 ||
      raw == -32768 ||
      sensorVoltage >= ALARM_THRESHOLD_V ||
      sensorVoltage < -0.02f ||
      sensorVoltage > 5.05f) {
    Serial.println(F("ERR"));
    return;
  }

  float nominalDistanceMm =
      65.0f - 6.0f * sensorVoltage;

  if (nominalDistanceMm < 35.0f ||
      nominalDistanceMm > 65.0f) {
    Serial.println(F("ERR"));
    return;
  }

  float sensorDisplayMm =
      CAL_SLOPE * nominalDistanceMm + CAL_OFFSET;

  if (sensorDisplayMm > -0.005f &&
      sensorDisplayMm < 0.005f) {
    sensorDisplayMm = 0.0f;
  }

  Serial.println(sensorDisplayMm, 2);
}

