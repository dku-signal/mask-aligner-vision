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
//   #X <letters>                     stop only those axes, e.g. "#X FBfbAud" (fine stage) -> "STOP <c> <n>" each
// X/x stops everything. Every line command it cancels first reports how far it actually went:
// "STOP <c> <steps|ms>" (n may be 0), then a bare "STOP".
// v2 (2026-11-24): the four STEP/DIR rails run AT THE SAME TIME (one 50us timer drives every active rail),
// and each DC pair (u/d, O/P, K/L) has its own timer, so the fine stage can align one wafer while the
// transfer arm fetches the next. Manual letters move only their own axis. No limit switches.
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
// 레일 STEP 상태 (v2: 레일 4개가 동시에 움직일 수 있다)
// Timer2: 50us마다 위상 반전 → 활성 레일 모두 같은 펄스 (약 10,000 스텝/초)
// ------------------------
const bool RAIL2_PAIR_INVERT = false;   // 이송부 양쪽 모터가 반대로 설치돼 있으면 true

const byte RAIL_COUNT = 4;              // 0: 정밀 X(F/B) 1: 정밀 Y(f/b) 2: 레일1(Q/W) 3: 레일2(E/R)
const byte RAIL_STEP[RAIL_COUNT] = {X_STEP_PIN, Y_STEP_PIN, RAIL1_STEP_PIN, RAIL2_STEP_PIN};
const byte RAIL_DIR[RAIL_COUNT]  = {X_DIR_PIN,  Y_DIR_PIN,  RAIL1_DIR_PIN,  RAIL2_DIR_PIN};

volatile bool railActive[RAIL_COUNT];
volatile bool railLowSent[RAIL_COUNT];
volatile long railLeft[RAIL_COUNT];     // -1 = 수동(무제한), >0 = 남은 스텝
volatile bool railFinished[RAIL_COUNT];
char railCmd[RAIL_COUNT];               // 진행 중인 줄 명령 글자 (0 = 없음/수동)
long railTotal[RAIL_COUNT];
volatile bool stepPhase = HIGH;

// DC 축 3쌍: 0: Z1(u/d) 1: Z2(O/P) 2: Z3(K/L)
const byte DC_COUNT = 3;
const byte DC_IN1[DC_COUNT] = {Z1_IN1_PIN, Z2_IN1_PIN, Z3_IN1_PIN};
const byte DC_IN2[DC_COUNT] = {Z1_IN2_PIN, Z2_IN2_PIN, Z3_IN2_PIN};
char dcCmd[DC_COUNT];                   // 진행 중인 줄 명령 글자 (0 = 없음/수동)
unsigned long dcStart[DC_COUNT], dcMs[DC_COUNT];

bool servoTurning = false;
unsigned long servoStart = 0, servoMs = 0;
bool thetaTurning = false;
unsigned long thetaStart = 0, thetaMs = 0;
const int THETA_TURN_SPEED = 30;       // 정밀 θ 연속회전 서보: 90±30
const int PREALIGN_TURN_SPEED = 30;    // 프리얼라인 연속회전 서보: 90±30

char lineBuf[40];
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
void stopAllMotion();
void handleCommand(char command);
void updateSensor();
void printSensorValue(int16_t raw);
bool takeWireTimeout();
void clearWireTimeout();

// ========================================
// Timer2 STEP 생성: 활성 레일 모두에 같은 펄스
// ========================================
ISR(TIMER2_COMPA_vect) {
  stepPhase = !stepPhase;
  bool any = false;
  for (byte i = 0; i < RAIL_COUNT; i++) {
    if (!railActive[i]) continue;
    any = true;
    digitalWrite(RAIL_STEP[i], stepPhase ? HIGH : LOW);
    if (!stepPhase) {
      railLowSent[i] = true;
    } else if (railLowSent[i]) {          // LOW→HIGH 한 번 = 1스텝
      railLowSent[i] = false;
      if (railLeft[i] > 0 && --railLeft[i] == 0) {
        railActive[i] = false;
        railFinished[i] = true;
      }
    }
  }
  if (!any) TIMSK2 &= ~_BV(OCIE2A);
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
// 레일 제어 (v2)
// ========================================
// 글자 → 레일 번호와 DIR. 반환 false면 레일 글자가 아님
bool railOf(char c, byte &rail, byte &dir) {
  switch (c) {
    case 'F': rail = 0; dir = LOW;  return true;
    case 'B': rail = 0; dir = HIGH; return true;
    case 'f': rail = 1; dir = LOW;  return true;
    case 'b': rail = 1; dir = HIGH; return true;
    case 'Q': rail = 2; dir = LOW;  return true;
    case 'W': rail = 2; dir = HIGH; return true;
    case 'E': rail = 3; dir = LOW;  return true;
    case 'R': rail = 3; dir = HIGH; return true;
  }
  return false;
}

int dcOf(char c, bool &upward) {
  switch (c) {
    case 'u': upward = true;  return 0;
    case 'd': upward = false; return 0;
    case 'O': upward = true;  return 1;
    case 'P': upward = false; return 1;
    case 'K': upward = true;  return 2;
    case 'L': upward = false; return 2;
  }
  return -1;
}

// 레일 하나 정지. 줄 명령이 진행 중이었으면 실제로 간 양을 "STOP <c> <n>"으로 알린다.
void stopRail(byte i) {
  long done = 0;
  bool pending;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    pending = railCmd[i] && !railFinished[i];
    if (pending && railLeft[i] > 0) done = railTotal[i] - railLeft[i];
    railActive[i] = false;
    railLowSent[i] = false;
    railLeft[i] = -1;
    digitalWrite(RAIL_STEP[i], HIGH);
  }
  if (pending) {
    Serial.print(F("STOP ")); Serial.print(railCmd[i]); Serial.print(' '); Serial.println(done);
    railCmd[i] = 0;
  }
}

// 레일 i를 dir로 시작. steps < 0 = 수동(무제한), 줄 명령이면 cmd 글자를 기록
void startRail(byte i, byte dir, long steps, char cmd) {
  stopRail(i);
  digitalWrite(RAIL_DIR[i], dir);
  delayMicroseconds(100);
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    railLeft[i] = steps;
    railFinished[i] = false;
    railLowSent[i] = false;
    railCmd[i] = cmd;
    railTotal[i] = steps;
    bool timerOn = TIMSK2 & _BV(OCIE2A);
    railActive[i] = true;
    if (!timerOn) {
      stepPhase = HIGH;
      TCNT2 = 0;
      TIFR2 = _BV(OCF2A);
      TIMSK2 |= _BV(OCIE2A);
    }
  }
}

// ========================================
// DC 축 (Z, Y 리니어): LOW/LOW = 정지
// ========================================
void stopZ(byte in1Pin, byte in2Pin) {
  digitalWrite(in1Pin, LOW);
  digitalWrite(in2Pin, LOW);
}

void stopDc(byte i) {
  if (dcCmd[i]) {
    Serial.print(F("STOP ")); Serial.print(dcCmd[i]); Serial.print(' '); Serial.println(millis() - dcStart[i]);
    dcCmd[i] = 0;
  }
  stopZ(DC_IN1[i], DC_IN2[i]);
}

void startDc(byte i, bool upward, char cmd, unsigned long ms) {
  stopDc(i);
  digitalWrite(DC_IN1[i], upward ? HIGH : LOW);
  digitalWrite(DC_IN2[i], upward ? LOW : HIGH);
  dcCmd[i] = cmd; dcStart[i] = millis(); dcMs[i] = ms;
}

void stopServoT(bool report) {
  if (!servoTurning) return;
  prealignServo.write(SERVO_STOP);
  servoTurning = false;
  if (report) { Serial.print(F("STOP t ")); Serial.println(millis() - servoStart); }
}

void stopServoA(bool report) {
  if (!thetaTurning) return;
  thetaServo.write(SERVO_STOP);
  thetaTurning = false;
  if (report) { Serial.print(F("STOP A ")); Serial.println(millis() - thetaStart); }
}

// 글자 목록에 해당하는 축만 정지 ("#X FBfbAud")
void stopLetters(const char *letters) {
  byte rail, dir; bool up;
  for (const char *p = letters; *p; p++) {
    if (railOf(*p, rail, dir)) stopRail(rail);
    else if (dcOf(*p, up) >= 0) stopDc(dcOf(*p, up));
    else if (*p == 'J' || *p == 'H') { stopRail(2); stopRail(3); }
    else if (*p == 't') stopServoT(true);
    else if (*p == 'A') stopServoA(true);
  }
}

void stopAllMotion() {
  bool pending = servoTurning || thetaTurning;
  for (byte i = 0; i < RAIL_COUNT; i++) pending |= railCmd[i] && !railFinished[i];
  for (byte i = 0; i < DC_COUNT; i++) pending |= dcCmd[i] != 0;
  for (byte i = 0; i < RAIL_COUNT; i++) stopRail(i);
  for (byte i = 0; i < DC_COUNT; i++) stopDc(i);
  stopServoT(true);
  stopServoA(true);
  if (pending) Serial.println(F("STOP"));
}

// ========================================
// 줄 명령
// ========================================
void runLine(char *line) {
  char c = line[0];
  long n = 0, dir = 0;
  byte rail, rdir; bool up;
  if (strcmp(line, "PING") == 0) { Serial.println(F("PONG")); return; }
  if (c == 'X' && line[1] == ' ') { stopLetters(line + 2); return; }
  if (c == 'M' && sscanf(line + 1, "%ld", &n) == 1) {
    magnet1On = n != 0;
    digitalWrite(MAGNET1_PIN, magnet1On ? HIGH : LOW);
    Serial.print(F("DONE M ")); Serial.println(magnet1On ? 1 : 0);
    return;
  }
  if (c == 't' && sscanf(line + 1, "%ld %ld", &n, &dir) == 2 && n > 0 && n <= 20000) {
    stopServoT(true);
    prealignServo.write(SERVO_STOP + (dir >= 0 ? PREALIGN_TURN_SPEED : -PREALIGN_TURN_SPEED));
    servoTurning = true; servoStart = millis(); servoMs = n;
    return;
  }
  if (c == 'A' && sscanf(line + 1, "%ld %ld", &n, &dir) == 2 && n > 0 && n <= 20000) {
    stopServoA(true);
    thetaServo.write(SERVO_STOP + (dir >= 0 ? THETA_TURN_SPEED : -THETA_TURN_SPEED));
    thetaTurning = true; thetaStart = millis(); thetaMs = n;
    return;
  }
  if (sscanf(line + 1, "%ld", &n) != 1 || n <= 0) { Serial.print(F("ERR ")); Serial.println(line); return; }
  if (railOf(c, rail, rdir)) {
    if (n > 2000000L) { Serial.println(F("ERR steps")); return; }
    startRail(rail, rdir, n, c);
  } else if (dcOf(c, up) >= 0) {
    if (n > 30000) { Serial.println(F("ERR ms")); return; }
    startDc(dcOf(c, up), up, c, n);
  } else {
    Serial.print(F("ERR ")); Serial.println(line);
  }
}

void serviceTimers() {
  for (byte i = 0; i < RAIL_COUNT; i++) {
    if (railFinished[i]) {
      char c; long total;
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        railFinished[i] = false;
        c = railCmd[i]; total = railTotal[i];
        railCmd[i] = 0;
        railLeft[i] = -1;
      }
      digitalWrite(RAIL_STEP[i], HIGH);
      if (c) { Serial.print(F("DONE ")); Serial.print(c); Serial.print(' '); Serial.println(total); }
    }
  }
  for (byte i = 0; i < DC_COUNT; i++) {
    if (dcCmd[i] && millis() - dcStart[i] >= dcMs[i]) {
      stopZ(DC_IN1[i], DC_IN2[i]);
      Serial.print(F("DONE ")); Serial.print(dcCmd[i]); Serial.print(' '); Serial.println(dcMs[i]);
      dcCmd[i] = 0;
    }
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
  // '#'로 시작하면 줄 명령, 아니면 한 글자 수동 명령
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
  updateSensor();
}

// ========================================
// 한 글자 수동 명령: 그 축만 움직인다 (누르는 동안, X로 정지)
// ========================================
void handleCommand(char command) {
  byte rail, dir; bool up;
  if (railOf(command, rail, dir)) { startRail(rail, dir, -1, 0); return; }
  int dc = dcOf(command, up);
  if (dc >= 0) {
    if (dcCmd[dc]) stopDc(dc);
    digitalWrite(DC_IN1[dc], up ? HIGH : LOW);
    digitalWrite(DC_IN2[dc], up ? LOW : HIGH);
    return;
  }
  switch (command) {
    case 'J':
    case 'H': {
      byte d = command == 'J' ? LOW : HIGH;
      startRail(2, d, -1, 0);
      startRail(3, RAIL2_PAIR_INVERT ? !d : d, -1, 0);
      break;
    }
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
      prealignServo.write(prealignAtB ? PREALIGN_ANGLE_B : PREALIGN_ANGLE_A);
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
      break;   // 줄바꿈 및 기타 문자 무시
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

