// Mask aligner + wafer loader, one Arduino Mega 2560 (SW protocol + HW team's manual letters)
//
// Merges firmware/xy_tb6560 (finite moves with replies, used by the Python app) with the
// HW team's Mega_All_Control sketch (2026-11-11, manual single-letter control + laser).
//
// Pins (HW team's Mega_All_Control, 2026-11-11; the Foyu axis is gone):
//   Aligner X  : STEP D48, DIR D49                Aligner Y : STEP D22, DIR D23
//   Rail 1     : STEP D52, DIR D53                Rail 2    : STEP D26, DIR D27   (TB6560, STEP idles HIGH)
//   Z1         : IN1 D2,  IN2 D3  ("original Z")  (DC H-bridge, LOW/LOW = stop)
//   Z2         : IN1 D34, IN2 D35                 Z3        : IN1 D38, IN2 D39
//   Theta      : servo D5                         Pre-align : servo D6
//   Magnet     : D7 (HIGH = ON)
//   Laser      : HG-C1050 voltage out -> 30k/30k divider -> ADS1115 A0 (I2C D20/D21, addr 0x48)
//
// Line commands (newline terminated). The first group is unchanged from xy_tb6560, so
// mask_aligner_optimized_v4.py, calibrate_xy.py and preflight.py work as before.
//   X <steps> / Y <steps>      move aligner axis, reply "MOVE X <n>" ... "DONE" (soft limit applies)
//   PING -> PONG   POS -> "POS X= Y= LIM= R1= R2="   ZERO   LIM <steps>   SPD <us>
//   T <id> <dir> <ms> [speed]  continuous theta servo: timed turn, reply "DONE T <id>"
//   P <us>                     hold a theta pulse width (servo test), reply "DONE P <us>"
//   s | STOP                   stop everything that moves, reply "STOP"
// Added for the loader / Z / sensor:
//   R1 <steps> / R2 <steps>    move rail 1 / rail 2, reply "MOVE R1 <n>" ... "DONE"
//   Z <mask> <dir> <ms>        run Z1 (1), Z2 (2), Z3 (4) or a sum of them; dir +1 = up; reply "DONE Z"
//   MAG 1 <0|1>                magnet off/on, reply "DONE MAG 1 <v>"
//   TA <deg> / PA <deg>        theta / pre-align angle (positional servo only), reply "DONE TA <deg>"
//   PT <id> <dir> <ms> [speed] pre-align continuous servo timed turn, reply "DONE PT <id>"
//   DIST                       one laser reading, reply "DIST <mm>" | "DIST ERR" | "DIST NOSENSOR"
//   STREAM <0|1>               print "D <mm>" every 200 ms while idle (HW manual use; app keeps it 0)
//
// HW team's manual letters (a single letter on its own line, as in their sketch):
//   F/B X jog   f/b Y jog   Q/W rail 1 jog   E/R rail 2 jog
//   u/d Z1 up/down   O/P Z2 up/down   K/L Z3 up/down   T/G theta 90/0   t pre-align 0/90   M magnet toggle
//   X or x stops all motion (magnets and servo angles are kept).
// Jogs run until a stop letter (s S x X) or a timeout: rails JOG_MAX_MS, Z ZJOG_MAX_MS.
//
// One thing moves at a time; every move checks for a stop letter between steps.

#include <EEPROM.h>
#include <Servo.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

#if !defined(__AVR_ATmega2560__)
#error "Select Arduino Mega 2560."
#endif

// --- servo type: set after the HW team confirms ------------------------------------
// false = continuous rotation (MG996R 360: angle commands are refused, timed T/PT only)
// true  = positional (0..180 deg: TA/PA and the T/G/t letters work)
const bool THETA_POSITIONAL = false;
const bool PREALIGN_POSITIONAL = false;

const byte X_STEP_PIN = 48;
const byte X_DIR_PIN = 49;
const byte Y_STEP_PIN = 22;
const byte Y_DIR_PIN = 23;
const byte R1_STEP_PIN = 52;
const byte R1_DIR_PIN = 53;
const byte R2_STEP_PIN = 26;
const byte R2_DIR_PIN = 27;

// Z H-bridges: {IN1, IN2}; Z command bit i drives Z_PINS[i].
const byte Z_COUNT = 3;
const byte Z_PINS[Z_COUNT][2] = {{2, 3}, {34, 35}, {38, 39}};
const unsigned long Z_MAX_MS = 15000;

const byte THETA_PIN = 5;
const byte PREALIGN_PIN = 6;
const int SERVO_NEUTRAL_US = 1500;  // stop for a continuous servo, 90 deg for a positional one
const int THETA_SPEED_US = 100;
const unsigned long THETA_MAX_MS = 250;
const unsigned long PREALIGN_MAX_MS = 2000;
Servo thetaServo;
Servo prealignServo;
bool thetaAtB = false;
bool prealignAtB = false;

const byte MAGNET_COUNT = 1;
const byte MAGNET_PINS[MAGNET_COUNT] = {7};
bool magnetOn[MAGNET_COUNT] = {false};

// Laser (HW team's calibration, 2026-11)
Adafruit_ADS1115 ads;
const uint8_t ADS_ADDRESS = 0x48;
const float DIVIDER_FACTOR = 2.0f;
const float ALARM_THRESHOLD_V = 5.10f;
const float CAL_SLOPE = -0.995459184f;
const float CAL_OFFSET = 49.671333329f;
const unsigned long STREAM_MS = 200;
bool adsReady = false;
bool streaming = false;
unsigned long lastStream = 0;

// Step timing (xy_tb6560): 100 us half-period, ramped from 400 us over 200 steps.
unsigned int halfPeriodUs = 100;
const unsigned int RAMP_START_US = 400;
const long RAMP_STEPS = 200;

long softLimit = 4000;                   // aligner X/Y only; EEPROM keeps the last LIM
const long MAX_MOVE_STEPS = 20000;       // aligner X/Y per command (the app clamps lower)
const long RAIL_MAX_MOVE_STEPS = 400000; // loader rails per command
const unsigned long JOG_MAX_MS = 3000;
const unsigned long ZJOG_MAX_MS = 10000;

long posX = 0, posY = 0, posR1 = 0, posR2 = 0;

char lineBuf[48];
byte lineLen = 0;
bool discardLine = false;

struct Axis {
  byte stepPin;
  byte dirPin;
  long *pos;
  const char *name;
  bool limited;
};

Axis axisX = {X_STEP_PIN, X_DIR_PIN, &posX, "X", true};
Axis axisY = {Y_STEP_PIN, Y_DIR_PIN, &posY, "Y", true};
Axis axisR1 = {R1_STEP_PIN, R1_DIR_PIN, &posR1, "R1", false};
Axis axisR2 = {R2_STEP_PIN, R2_DIR_PIN, &posR2, "R2", false};

// Same layout and magic as xy_tb6560: flashing this keeps the saved X/Y/LIM.
struct Saved {
  long magic;
  long x;
  long y;
  long limit;
};
const long SAVE_MAGIC = 0x58595442L;  // "XYTB"

void savePosition() {
  Saved s = {SAVE_MAGIC, posX, posY, softLimit};
  EEPROM.put(0, s);
}

void loadPosition() {
  Saved s;
  EEPROM.get(0, s);
  if (s.magic == SAVE_MAGIC && s.limit > 0) {
    posX = s.x;
    posY = s.y;
    softLimit = s.limit;
  }
}

// ---------------------------------------------------------------- stop helpers

void idleRails() {
  digitalWrite(X_STEP_PIN, HIGH);
  digitalWrite(Y_STEP_PIN, HIGH);
  digitalWrite(R1_STEP_PIN, HIGH);
  digitalWrite(R2_STEP_PIN, HIGH);
}

void zStop() {
  for (byte i = 0; i < Z_COUNT; i++) {
    digitalWrite(Z_PINS[i][0], LOW);
    digitalWrite(Z_PINS[i][1], LOW);
  }
}

// A continuous servo creeps at "neutral" 1500 us (2026-11-05: the disc turned ~2 deg and
// moved the wafer key ~5 px in 70 s after ALIGN FINISH). With no pulses at all it stops,
// so a continuous servo is only attached while it is being driven.
void servoRelease(Servo &servo, bool positional) {
  if (positional) return;
  servo.writeMicroseconds(SERVO_NEUTRAL_US);
  servo.detach();
}

void servoDrive(Servo &servo, byte pin, int us) {
  servo.writeMicroseconds(us);  // set before attach: the first pulse is already this width
  if (!servo.attached()) servo.attach(pin);
}

void continuousServosStop() {
  servoRelease(thetaServo, THETA_POSITIONAL);
  servoRelease(prealignServo, PREALIGN_POSITIONAL);
}

// Stops motion only: magnets stay as they are, positional servos keep their angle.
void stopAll() {
  idleRails();
  zStop();
  continuousServosStop();
}

bool isStopChar(char c) {
  return c == 's' || c == 'S' || c == 'x' || c == 'X';
}

// True if a stop letter arrived (consumed). Other input waits until the move ends.
bool stopRequested() {
  while (Serial.available() > 0) {
    char c = Serial.peek();
    if (isStopChar(c)) {
      Serial.read();
      discardLine = true;  // swallow the rest of "STOP"
      return true;
    }
    if (c == '\n' || c == '\r') {
      Serial.read();
      continue;
    }
    return false;
  }
  return false;
}

// ---------------------------------------------------------------- step rails

bool runSteps(Axis &a, long steps) {
  int dir = steps > 0 ? 1 : -1;
  long n = labs(steps);
  digitalWrite(a.dirPin, dir > 0 ? LOW : HIGH);
  delayMicroseconds(100);
  for (long i = 0; i < n; i++) {
    if (stopRequested()) {
      idleRails();
      return false;
    }
    long fromEdge = min(i, n - 1 - i);
    unsigned int us = halfPeriodUs;
    if (fromEdge < RAMP_STEPS && RAMP_START_US > halfPeriodUs) {
      us = RAMP_START_US - (unsigned int)((long)(RAMP_START_US - halfPeriodUs) * fromEdge / RAMP_STEPS);
    }
    digitalWrite(a.stepPin, LOW);
    delayMicroseconds(us);
    digitalWrite(a.stepPin, HIGH);
    delayMicroseconds(us);
    *a.pos += dir;
  }
  return true;
}

void moveAxis(Axis &a, long steps) {
  long maxSteps = a.limited ? MAX_MOVE_STEPS : RAIL_MAX_MOVE_STEPS;
  if (steps == 0 || labs(steps) > maxSteps) {
    Serial.print("ERR ");
    Serial.print(a.name);
    Serial.println(" range");
    return;
  }
  long target = *a.pos + steps;
  if (a.limited && (target > softLimit || target < -softLimit)) {
    Serial.print("ERR ");
    Serial.print(a.name);
    Serial.print(" limit pos=");
    Serial.print(*a.pos);
    Serial.print(" lim=");
    Serial.println(softLimit);
    return;
  }
  Serial.print("MOVE ");
  Serial.print(a.name);
  Serial.print(" ");
  Serial.println(steps);
  bool completed = runSteps(a, steps);
  if (a.limited) savePosition();
  Serial.println(completed ? "DONE" : "STOP");
}

void jog(Axis &a, int dir) {
  Serial.print(a.name);
  Serial.println(dir > 0 ? " Forward" : " Backward");
  unsigned long start = millis();
  digitalWrite(a.dirPin, dir > 0 ? LOW : HIGH);
  delayMicroseconds(100);
  while (true) {
    if (stopRequested()) {
      Serial.println("STOP");
      break;
    }
    if (millis() - start > JOG_MAX_MS) {
      Serial.println("STOP jog timeout");
      break;
    }
    long next = *a.pos + dir;
    if (a.limited && (next > softLimit || next < -softLimit)) {
      Serial.println("STOP soft limit");
      break;
    }
    digitalWrite(a.stepPin, LOW);
    delayMicroseconds(halfPeriodUs);
    digitalWrite(a.stepPin, HIGH);
    delayMicroseconds(halfPeriodUs);
    *a.pos = next;
  }
  idleRails();
  if (a.limited) savePosition();
}

// ---------------------------------------------------------------- Z (DC actuators)

void zDrive(byte which, int dir) {
  // which: bit 0 = Z1, bit 1 = Z2, bit 2 = Z3. dir +1 = up (IN1 HIGH), -1 = down.
  for (byte i = 0; i < Z_COUNT; i++) {
    if (!(which & (1 << i))) continue;
    digitalWrite(Z_PINS[i][0], dir > 0 ? HIGH : LOW);
    digitalWrite(Z_PINS[i][1], dir > 0 ? LOW : HIGH);
  }
}

// Returns false if stopped early.
bool runZ(byte which, int dir, unsigned long ms) {
  zStop();
  zDrive(which, dir);
  unsigned long start = millis();
  while (millis() - start < ms) {
    if (stopRequested()) {
      zStop();
      return false;
    }
  }
  zStop();
  return true;
}

void moveZ(char *line) {
  long which = 0, ms = 0;
  int dir = 0;
  char extra;
  if (sscanf(line, "Z %ld %d %ld %c", &which, &dir, &ms, &extra) != 3 || which < 1 || which >= (1 << Z_COUNT) ||
      (dir != 1 && dir != -1) || ms < 1 || ms > (long)Z_MAX_MS) {
    Serial.println("ERR Z range");
    return;
  }
  bool completed = runZ((byte)which, dir, (unsigned long)ms);
  Serial.println(completed ? "DONE Z" : "STOP");
}

void jogZ(byte which, int dir) {
  Serial.print(which == 1 ? "Z1" : which == 2 ? "Z2" : "Z3");
  Serial.println(dir > 0 ? " Up" : " Down");
  bool completed = runZ(which, dir, ZJOG_MAX_MS);
  Serial.println(completed ? "STOP jog timeout" : "STOP");
}

// ---------------------------------------------------------------- servos

// Timed turn of a continuous servo, then neutral. Returns false if stopped.
bool timedTurn(Servo &servo, byte pin, int dir, unsigned long ms, int speed) {
  servoDrive(servo, pin, SERVO_NEUTRAL_US + dir * speed);
  unsigned long start = millis();
  while (millis() - start < ms) {
    if (stopRequested()) {
      servoRelease(servo, false);
      return false;
    }
  }
  servoRelease(servo, false);
  return true;
}

// "T id dir ms [speed]" (prefix "T") or "PT id dir ms [speed]" (prefix "PT").
void turnCommand(char *line, bool prealign) {
  long id = 0, ms = 0, speed = THETA_SPEED_US;
  int dir = 0;
  char extra;
  const char *fmt = prealign ? "PT %ld %d %ld %ld %c" : "T %ld %d %ld %ld %c";
  const char *name = prealign ? "PT" : "T";
  bool positional = prealign ? PREALIGN_POSITIONAL : THETA_POSITIONAL;
  unsigned long maxMs = prealign ? PREALIGN_MAX_MS : THETA_MAX_MS;
  int fields = sscanf(line, fmt, &id, &dir, &ms, &speed, &extra);
  if (positional) {
    Serial.print("ERR ");
    Serial.print(name);
    Serial.println(" positional servo: use angle");
    return;
  }
  if ((fields != 3 && fields != 4) || id <= 0 || (dir != 1 && dir != -1) ||
      ms < 20 || ms > (long)maxMs || speed < 50 || speed > 500) {
    continuousServosStop();
    Serial.print("ERR ");
    Serial.print(name);
    Serial.println(" range");
    return;
  }
  bool completed = timedTurn(prealign ? prealignServo : thetaServo, prealign ? PREALIGN_PIN : THETA_PIN,
                             dir, (unsigned long)ms, (int)speed);
  if (!completed) {
    Serial.println("STOP");
    return;
  }
  Serial.print("DONE ");
  Serial.print(name);
  Serial.print(" ");
  Serial.println(id);
}

void angleCommand(Servo &servo, bool positional, const char *name, long deg) {
  if (!positional) {
    Serial.print("ERR ");
    Serial.print(name);
    Serial.println(" continuous servo: use timed turn");
    return;
  }
  if (deg < 0 || deg > 180) {
    Serial.print("ERR ");
    Serial.print(name);
    Serial.println(" range");
    return;
  }
  servo.write((int)deg);
  Serial.print("DONE ");
  Serial.print(name);
  Serial.print(" ");
  Serial.println(deg);
}

// ---------------------------------------------------------------- magnets

void setMagnet(byte i, bool on) {
  magnetOn[i] = on;
  digitalWrite(MAGNET_PINS[i], on ? HIGH : LOW);
}

// ---------------------------------------------------------------- laser

// Distance in mm as the sensor display shows it; returns false on a bad reading.
bool readLaser(float &mm) {
  if (!adsReady) return false;
#if defined(WIRE_HAS_TIMEOUT)
  Wire.clearWireTimeoutFlag();
#endif
  int16_t raw = ads.readADC_SingleEnded(0);
#if defined(WIRE_HAS_TIMEOUT)
  if (Wire.getWireTimeoutFlag()) {
    Wire.clearWireTimeoutFlag();
    return false;
  }
#endif
  float sensorVoltage = ads.computeVolts(raw) * DIVIDER_FACTOR;
  if (raw == 32767 || raw == -32768 || sensorVoltage >= ALARM_THRESHOLD_V ||
      sensorVoltage < -0.02f || sensorVoltage > 5.05f) {
    return false;
  }
  float nominal = 65.0f - 6.0f * sensorVoltage;
  if (nominal < 35.0f || nominal > 65.0f) return false;
  mm = CAL_SLOPE * nominal + CAL_OFFSET;
  if (mm > -0.005f && mm < 0.005f) mm = 0.0f;
  return true;
}

void printDistance(const char *prefix) {
  float mm;
  Serial.print(prefix);
  if (!adsReady) {
    Serial.println("NOSENSOR");
  } else if (readLaser(mm)) {
    Serial.println(mm, 2);
  } else {
    Serial.println("ERR");
  }
}

// ---------------------------------------------------------------- commands

// HW team's single letters. Returns false if c is not one of them.
bool manualLetter(char c) {
  switch (c) {
    case 'F': jog(axisX, 1); return true;
    case 'B': jog(axisX, -1); return true;
    case 'f': jog(axisY, 1); return true;
    case 'b': jog(axisY, -1); return true;
    case 'Q': jog(axisR1, 1); return true;
    case 'W': jog(axisR1, -1); return true;
    case 'E': jog(axisR2, 1); return true;
    case 'R': jog(axisR2, -1); return true;
    case 'u': jogZ(1, 1); return true;
    case 'd': jogZ(1, -1); return true;
    case 'O': jogZ(2, 1); return true;
    case 'P': jogZ(2, -1); return true;
    case 'K': jogZ(4, 1); return true;
    case 'L': jogZ(4, -1); return true;
    case 'T':
    case 'G':
      if (!THETA_POSITIONAL) {
        Serial.println("ERR theta is continuous: use T <id> <dir> <ms>");
      } else {
        thetaAtB = (c == 'T');
        thetaServo.write(thetaAtB ? 90 : 0);
        Serial.println(thetaAtB ? "THETA 90" : "THETA 0");
      }
      return true;
    case 't':
      if (!PREALIGN_POSITIONAL) {
        Serial.println("ERR pre-align is continuous: use PT <id> <dir> <ms>");
      } else {
        prealignAtB = !prealignAtB;
        prealignServo.write(prealignAtB ? 90 : 0);
        Serial.println(prealignAtB ? "PREALIGN 90" : "PREALIGN 0");
      }
      return true;
    case 'M':
      setMagnet(0, !magnetOn[0]);
      Serial.println(magnetOn[0] ? "MAG 1 ON" : "MAG 1 OFF");
      return true;
    case 'X':
    case 'x':
    case 's':
      stopAll();
      Serial.println("STOP");
      return true;
  }
  return false;
}

bool isManualLetter(char c) {
  return strchr("FBfbQWERudOPKLTGtMXxs", c) != NULL;
}

void processLine(char *line) {
  long n = 0, a = 0, b = 0;
  char extra;
  if (strcmp(line, "STOP") == 0) {
    stopAll();
    Serial.println("STOP");
  } else if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(line, "POS") == 0) {
    Serial.print("POS X=");
    Serial.print(posX);
    Serial.print(" Y=");
    Serial.print(posY);
    Serial.print(" LIM=");
    Serial.print(softLimit);
    Serial.print(" R1=");
    Serial.print(posR1);
    Serial.print(" R2=");
    Serial.println(posR2);
  } else if (strcmp(line, "ZERO") == 0) {
    posX = posY = 0;
    savePosition();
    Serial.println("DONE");
  } else if (sscanf(line, "LIM %ld %c", &n, &extra) == 1 && n > 0) {
    softLimit = n;
    savePosition();
    Serial.println("DONE");
  } else if (sscanf(line, "SPD %ld %c", &n, &extra) == 1 && n >= 100 && n <= 5000) {
    halfPeriodUs = (unsigned int)n;
    Serial.println("DONE");
  } else if (sscanf(line, "X %ld %c", &n, &extra) == 1) {
    moveAxis(axisX, n);
  } else if (sscanf(line, "Y %ld %c", &n, &extra) == 1) {
    moveAxis(axisY, n);
  } else if (sscanf(line, "R1 %ld %c", &n, &extra) == 1) {
    moveAxis(axisR1, n);
  } else if (sscanf(line, "R2 %ld %c", &n, &extra) == 1) {
    moveAxis(axisR2, n);
  } else if (line[0] == 'Z' && line[1] == ' ') {
    moveZ(line);
  } else if (sscanf(line, "MAG %ld %ld %c", &a, &b, &extra) == 2 && a >= 1 && a <= MAGNET_COUNT && (b == 0 || b == 1)) {
    setMagnet((byte)(a - 1), b == 1);
    Serial.print("DONE MAG ");
    Serial.print(a);
    Serial.print(" ");
    Serial.println(b);
  } else if (strcmp(line, "DIST") == 0) {
    printDistance("DIST ");
  } else if (sscanf(line, "STREAM %ld %c", &n, &extra) == 1 && (n == 0 || n == 1)) {
    streaming = (n == 1);
    Serial.println("DONE");
  } else if (sscanf(line, "TA %ld %c", &n, &extra) == 1) {
    angleCommand(thetaServo, THETA_POSITIONAL, "TA", n);
  } else if (sscanf(line, "PA %ld %c", &n, &extra) == 1) {
    angleCommand(prealignServo, PREALIGN_POSITIONAL, "PA", n);
  } else if (line[0] == 'P' && line[1] == 'T' && line[2] == ' ') {
    turnCommand(line, true);
  } else if (line[0] == 'T' && line[1] == ' ') {
    turnCommand(line, false);
  } else if (sscanf(line, "P %ld %c", &n, &extra) == 1) {
    if (n >= 900 && n <= 2100) {
      servoDrive(thetaServo, THETA_PIN, (int)n);  // held until a stop (servo test)
      Serial.print("DONE P ");
      Serial.println(n);
    } else {
      Serial.println("ERR P range");
    }
  } else if (line[0] != 0) {
    Serial.print("ERR unknown ");
    Serial.println(line);
  }
}

void setup() {
  // Every output at its safe level before it becomes an output.
  idleRails();
  digitalWrite(X_DIR_PIN, LOW);
  digitalWrite(Y_DIR_PIN, LOW);
  digitalWrite(R1_DIR_PIN, LOW);
  digitalWrite(R2_DIR_PIN, LOW);
  const byte outputs[] = {X_STEP_PIN, X_DIR_PIN, Y_STEP_PIN, Y_DIR_PIN,
                          R1_STEP_PIN, R1_DIR_PIN, R2_STEP_PIN, R2_DIR_PIN};
  zStop();
  for (byte i = 0; i < MAGNET_COUNT; i++) digitalWrite(MAGNET_PINS[i], LOW);
  for (byte i = 0; i < sizeof(outputs); i++) pinMode(outputs[i], OUTPUT);
  for (byte i = 0; i < Z_COUNT; i++) {
    pinMode(Z_PINS[i][0], OUTPUT);
    pinMode(Z_PINS[i][1], OUTPUT);
  }
  for (byte i = 0; i < MAGNET_COUNT; i++) pinMode(MAGNET_PINS[i], OUTPUT);

  loadPosition();

  // Continuous servos get no pulses until driven (no creep). A positional one is attached
  // at neutral = 90 deg so its angle commands work.
  if (THETA_POSITIONAL) servoDrive(thetaServo, THETA_PIN, SERVO_NEUTRAL_US);
  if (PREALIGN_POSITIONAL) servoDrive(prealignServo, PREALIGN_PIN, SERVO_NEUTRAL_US);

  Serial.begin(9600);

  Wire.begin();
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(25000, true);
#endif
  adsReady = ads.begin(ADS_ADDRESS, &Wire);
  if (adsReady) {
    ads.setGain(GAIN_ONE);
    ads.setDataRate(RATE_ADS1115_128SPS);
  }

  Serial.print("ALIGNER MEGA READY pos X=");
  Serial.print(posX);
  Serial.print(" Y=");
  Serial.print(posY);
  Serial.print(" LIM=");
  Serial.print(softLimit);
  Serial.println(adsReady ? " laser=ok" : " laser=none");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (!discardLine) {
        lineBuf[lineLen] = 0;
        processLine(lineBuf);
      }
      lineLen = 0;
      discardLine = false;
      continue;
    }
    if (discardLine) continue;
    // A manual letter acts on its own; a letter followed by more text is a line command.
    if (lineLen == 0 && isManualLetter(c)) {
      // At 9600 baud the next byte needs ~1 ms; wait so "PING", "X 100" are not cut short.
      unsigned long waitStart = micros();
      while (!Serial.available() && micros() - waitStart < 3000) {
      }
      int next = Serial.peek();
      if (next != ' ' && !(next >= 0 && isAlphaNumeric(next))) {
        discardLine = true;
        manualLetter(c);
        continue;
      }
    }
    if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      discardLine = true;
      Serial.println("ERR line too long");
    }
  }

  if (streaming && millis() - lastStream >= STREAM_MS) {
    lastStream = millis();
    printDistance("D ");
  }
}
