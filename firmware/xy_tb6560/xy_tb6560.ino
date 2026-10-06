// TB6560 X/Y stage for the Python mask aligner (Arduino Mega)
//
// Pins and pulse polarity follow the team's TB6560 jog sketch:
//   Y : STEP D8,  DIR D9
//   X : STEP D10, DIR D11
//   STEP idles HIGH (common-anode opto input); one LOW->HIGH cycle = one step.
//   Positive steps = "Forward" = DIR LOW.
//
// Every move is a finite step count. There is no limit switch, so the
// position is counted in steps and moves that would leave [-softLimit, +softLimit]
// are refused. The count and limit live in EEPROM: opening the serial port resets
// the Mega, and a reset must not re-centre the limits. Centre the stage once
// (F/B/f/b jog) and send ZERO.
//
// Line commands (newline terminated), compatible with mask_aligner_optimized_v4.py:
//   X <steps> / Y <steps>  move, reply "MOVE X <n>" ... "DONE"
//   s | S | STOP           stop now, reply "STOP"
//   PING                   reply "PONG"
//   POS                    reply "POS X=<x> Y=<y> LIM=<l>"
//   ZERO                   current position becomes 0, reply "DONE"
//   LIM <steps>            set soft limit (half travel), reply "DONE"
//   SPD <us>               set step half-period in us (min 100), reply "DONE"
// Manual jog (team sketch letters), bounded by the soft limit and JOG_MAX_MS:
//   F / B = X forward / backward,  f / b = Y forward / backward,  S = stop
// T ... (theta) is answered "ERR T disabled": no theta motor on this build.

#include <EEPROM.h>

const byte Y_STEP_PIN = 8;
const byte Y_DIR_PIN = 9;
const byte X_STEP_PIN = 10;
const byte X_DIR_PIN = 11;

// Team sketch used 100 us half-period. Moves ramp down to it from RAMP_START_US
// so the first steps are not lost.
unsigned int halfPeriodUs = 100;
const unsigned int RAMP_START_US = 400;
const long RAMP_STEPS = 200;

// Conservative until steps/mm is calibrated; raise with LIM once the travel is known.
long softLimit = 4000;
const long MAX_MOVE_STEPS = 20000;
const unsigned long JOG_MAX_MS = 3000;

long posX = 0;
long posY = 0;

char lineBuf[48];
byte lineLen = 0;
bool discardLine = false;

struct Axis {
  byte stepPin;
  byte dirPin;
  long *pos;
  char name;
};

Axis axisX = {X_STEP_PIN, X_DIR_PIN, &posX, 'X'};
Axis axisY = {Y_STEP_PIN, Y_DIR_PIN, &posY, 'Y'};

struct Saved {
  long magic;
  long x;
  long y;
  long limit;
};
const long SAVE_MAGIC = 0x58595442L;  // "XYTB"

void savePosition() {
  Saved s = {SAVE_MAGIC, posX, posY, softLimit};
  EEPROM.put(0, s);  // put() skips unchanged bytes
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

void idlePins() {
  digitalWrite(X_STEP_PIN, HIGH);
  digitalWrite(Y_STEP_PIN, HIGH);
}

// Returns true if a stop character arrived (it is consumed).
bool stopRequested() {
  while (Serial.available() > 0) {
    char c = Serial.peek();
    if (c == 's' || c == 'S') {
      Serial.read();
      discardLine = true;  // swallow the rest of "STOP"
      return true;
    }
    if (c == '\n' || c == '\r') {
      Serial.read();
      continue;
    }
    return false;  // leave other input for after the move
  }
  return false;
}

// Moves |steps| in the sign's direction. Returns false if stopped early.
bool runSteps(Axis &a, long steps) {
  int dir = steps > 0 ? 1 : -1;
  long n = labs(steps);
  digitalWrite(a.dirPin, dir > 0 ? LOW : HIGH);
  delayMicroseconds(100);
  for (long i = 0; i < n; i++) {
    if (stopRequested()) {
      idlePins();
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
  if (steps == 0 || labs(steps) > MAX_MOVE_STEPS) {
    Serial.print("ERR ");
    Serial.print(a.name);
    Serial.println(" range");
    return;
  }
  long target = *a.pos + steps;
  if (target > softLimit || target < -softLimit) {
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
  savePosition();
  Serial.println(completed ? "DONE" : "STOP");
}

// Jog until stop, soft limit or JOG_MAX_MS, one step at a time.
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
    if (next > softLimit || next < -softLimit) {
      Serial.println("STOP soft limit");
      break;
    }
    digitalWrite(a.stepPin, LOW);
    delayMicroseconds(halfPeriodUs);
    digitalWrite(a.stepPin, HIGH);
    delayMicroseconds(halfPeriodUs);
    *a.pos = next;
  }
  idlePins();
  savePosition();
}

void processLine(char *line) {
  long n = 0;
  char extra;
  if (strcmp(line, "STOP") == 0) {
    Serial.println("STOP");
  } else if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(line, "POS") == 0) {
    Serial.print("POS X=");
    Serial.print(posX);
    Serial.print(" Y=");
    Serial.print(posY);
    Serial.print(" LIM=");
    Serial.println(softLimit);
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
  } else if (line[0] == 'T' && (line[1] == ' ' || line[1] == 0)) {
    Serial.println("ERR T disabled");
  } else if (line[0] != 0) {
    Serial.print("ERR unknown ");
    Serial.println(line);
  }
}

void setup() {
  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT);
  pinMode(Y_DIR_PIN, OUTPUT);
  idlePins();
  digitalWrite(X_DIR_PIN, LOW);
  digitalWrite(Y_DIR_PIN, LOW);
  loadPosition();
  Serial.begin(9600);
  Serial.print("XY TB6560 READY pos X=");
  Serial.print(posX);
  Serial.print(" Y=");
  Serial.print(posY);
  Serial.print(" LIM=");
  Serial.println(softLimit);
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
    if (discardLine) {
      continue;
    }
    // Team-sketch single letters act immediately, with or without a line ending.
    if (lineLen == 0 && (c == 'F' || c == 'B' || c == 'f' || c == 'b' || c == 's' || c == 'S')) {
      // At 9600 baud the next byte needs ~1 ms; wait so "SPD"/"STOP" are not cut short.
      unsigned long waitStart = micros();
      while (!Serial.available() && micros() - waitStart < 3000) {
      }
      if (Serial.peek() == ' ' || isAlphaNumeric(Serial.peek())) {
        // Part of a longer word such as "STOP" or a typo; parse as a line.
      } else {
        discardLine = true;
        if (c == 'F') jog(axisX, 1);
        else if (c == 'B') jog(axisX, -1);
        else if (c == 'f') jog(axisY, 1);
        else if (c == 'b') jog(axisY, -1);
        else Serial.println("STOP");
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
}
