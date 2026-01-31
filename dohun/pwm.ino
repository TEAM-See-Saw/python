// ==========================================
// [1] 통합 설정 (주행 + 페일세이프)
// ==========================================
// --- 주행 설정 ---
const int CENTER_VAL = 570;
const int LIMIT_MIN = 480; 
const int LIMIT_MAX = 680; 
const int STEER_MIN_SPEED = 90; 
const int STEER_MAX_SPEED = 200; 
const int TOLERANCE = 15;

// --- 페일세이프 설정 ---
const unsigned long FAILSAFE_TIMEOUT = 1000; // 1초 동안 명령 없으면 정지
unsigned long lastCmdTime = 0;

// --- 핀 설정 (모터 & 서보) ---
const int R_IN1 = 3; 
const int R_IN2 = 4;
const int L_IN1 = 11; 
const int L_IN2 = 10; 
const int L_PWM_POWER = 12;
const int POT_PIN = A0;
const int POT_VCC_PIN = 13;
const int STR_IN1 = 8; 
const int STR_IN2 = 9; 
const int STR_PWM = 5;

// --- 제어 변수 ---
int targetAngle = CENTER_VAL;
bool autoSteering = false;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(10);

  pinMode(R_IN1, OUTPUT); 
  pinMode(R_IN2, OUTPUT);
  pinMode(L_IN1, OUTPUT); 
  pinMode(L_IN2, OUTPUT); 
  pinMode(L_PWM_POWER, OUTPUT);
  digitalWrite(L_PWM_POWER, HIGH);

  pinMode(POT_VCC_PIN, OUTPUT);
  digitalWrite(POT_VCC_PIN, HIGH);

  pinMode(STR_IN1, OUTPUT); 
  pinMode(STR_IN2, OUTPUT); 
  pinMode(STR_PWM, OUTPUT);

  lastCmdTime = millis();
  Serial.println("Ready: Drive + Failsafe");
}

void loop() {
  // 1. 파이썬 명령 수신 (주행 제어)
  if (Serial.available() > 0) {
    lastCmdTime = millis();

    String data = Serial.readStringUntil('\n');
    if (data.length() < 3) return; // 최소 "S,0" 형태 방어

    char cmd = data.charAt(0);
    int val = data.substring(2).toInt();

    if (cmd == 'S') {
      targetAngle = constrain(val, LIMIT_MIN, LIMIT_MAX);
      autoSteering = true;
    }
    else if (cmd == 'D') {
      moveRearMotors(val);
    }
  }

  // 1.5 페일세이프
  if (millis() - lastCmdTime > FAILSAFE_TIMEOUT) {
    stopRearMotors();
    stopSteering();
    autoSteering = false;
  }
  else {
    // 2. 자동 조향
    if (autoSteering) {
      int currentPot = analogRead(POT_PIN);
      int error = targetAngle - currentPot;

      if (abs(error) <= TOLERANCE) {
        stopSteering();
      } else {
        int speed = map(abs(error), 0, 100, STEER_MIN_SPEED, STEER_MAX_SPEED);
        speed = constrain(speed, STEER_MIN_SPEED, STEER_MAX_SPEED);
        if (error > 0) moveSteering(speed, true);
        else moveSteering(speed, false);
      }
    }
  }
}

// --- 유틸리티 함수들 ---

void moveSteering(int speed, bool increaseValue) {
  analogWrite(STR_PWM, speed);
  if (increaseValue) {
    digitalWrite(STR_IN1, LOW); 
    digitalWrite(STR_IN2, HIGH);
  } else {
    digitalWrite(STR_IN1, HIGH); 
    digitalWrite(STR_IN2, LOW);
  }
}

void stopSteering() { 
  analogWrite(STR_PWM, 0); 
  digitalWrite(STR_IN1, LOW); 
  digitalWrite(STR_IN2, LOW); 
}

void stopRearMotors() { 
  analogWrite(R_IN1, 0); 
  analogWrite(R_IN2, 0); 
  analogWrite(L_IN1, 0); 
  analogWrite(L_IN2, 0); 
}

void moveRearMotors(int speed) {
  if (speed == 0) { 
    stopRearMotors(); 
    return; 
  }
  if (speed > 0) {
    speed = constrain(speed, 0, 255);
    analogWrite(R_IN1, speed); 
    analogWrite(R_IN2, 0);
    analogWrite(L_IN1, speed); 
    analogWrite(L_IN2, 0);
  } else {
    int revSpeed = constrain(abs(speed), 0, 255);
    analogWrite(R_IN1, 0); 
    analogWrite(R_IN2, revSpeed);
    analogWrite(L_IN1, 0); 
    analogWrite(L_IN2, revSpeed);
  }
}
