#include <Arduino.h>
#include <Wire.h>
#include <MAX30105.h>
#include "esp_sleep.h"

MAX30105 particleSensor;

// --- Pin Configuration (ESP32-C3 Mini) ---
constexpr int BUTTON_PIN = 1;
constexpr int LED_PIN = 10;

// --- CRT State Machine ---
enum class AppState {
  IDLE,
  LED_BLINK,
  LED_SOLID,
  MEASURING,
  DONE
};

AppState currentState = AppState::IDLE;

// --- Timing Variables ---
unsigned long stateStartTime = 0;
unsigned long refillStartTime = 0;

// --- LED Blink Timing ---
constexpr unsigned long BLINK_DURATION_MS = 3000;
constexpr unsigned long BLINK_INTERVAL_MS = 250;
constexpr unsigned long SOLID_DURATION_MS = 5000;

// --- Baseline (Reference) Capture ---
float baselineIR = 0.0f;
int baselineSamples = 0;

// --- Recovery Measurement Variables ---
float smoothedIR = 0.0f;
float previousSmoothedIR = 0.0f;
int stableCount = 0;

// --- Algorithm Tuning Parameters ---
constexpr float EMA_ALPHA = 0.05f;
constexpr float STABILITY_THRESHOLD = 15.0f;
constexpr int REQUIRED_STABLE_SAMPLES = 50;

// --- Button Debounce ---
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
constexpr unsigned long DEBOUNCE_DELAY = 50;

void resetMeasurementState() {
  baselineIR = 0.0f;
  baselineSamples = 0;
  smoothedIR = 0.0f;
  previousSmoothedIR = 0.0f;
  stableCount = 0;
}

void transitionToState(AppState nextState) {
  currentState = nextState;
  stateStartTime = millis();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin();

  Serial.println("Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("FAILED! Check wiring.");
    while (1) {
      delay(1000);
    }
  }
  Serial.println("SUCCESS.");

  const byte ledBrightness = 60;
  const byte sampleAverage = 1;
  const byte ledMode = 2;
  const byte sampleRate = 100;
  const int pulseWidth = 411;
  const int adcRange = 4096;
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);

  Serial.println("\n========================================");
  Serial.println("   CAPILLARY REFILL TEST (Serial Mode)");
  Serial.println("========================================");
  Serial.println("Press the BUTTON to begin the test.");
  Serial.println("Results sent via USB Serial to Raspberry Pi.");
  Serial.println("========================================\n");
}

void loop() {
  particleSensor.check();
  const bool buttonReading = digitalRead(BUTTON_PIN);

  switch (currentState) {
    case AppState::IDLE: {
      if (lastButtonState == HIGH && buttonReading == LOW) {
        lastDebounceTime = millis();
      }
      if (lastButtonState == LOW && buttonReading == LOW &&
          (millis() - lastDebounceTime > DEBOUNCE_DELAY)) {
        Serial.println("\n[ BUTTON CONFIRMED - Place finger lightly on sensor ]");
        Serial.println("LED blinking for 3 seconds...");
        digitalWrite(LED_PIN, HIGH);
        delay(200);
        digitalWrite(LED_PIN, LOW);
        delay(200);
        transitionToState(AppState::LED_BLINK);
        resetMeasurementState();
      }
      break;
    }

    case AppState::LED_BLINK: {
      const unsigned long elapsed = millis() - stateStartTime;

      if ((elapsed / BLINK_INTERVAL_MS) % 2 == 0) {
        digitalWrite(LED_PIN, HIGH);
      } else {
        digitalWrite(LED_PIN, LOW);
      }

      while (particleSensor.available()) {
        const uint32_t ir = particleSensor.getFIFOIR();
        if (ir > 50000) {
          if (smoothedIR == 0.0f) {
            smoothedIR = static_cast<float>(ir);
          }
          smoothedIR = (EMA_ALPHA * static_cast<float>(ir)) + ((1.0f - EMA_ALPHA) * smoothedIR);
          baselineIR += smoothedIR;
          baselineSamples++;
        }
        particleSensor.nextSample();
      }

      if (elapsed >= BLINK_DURATION_MS) {
        if (baselineSamples > 0) {
          baselineIR /= baselineSamples;
          Serial.print("Baseline captured (");
          Serial.print(baselineSamples);
          Serial.print(" samples): ");
          Serial.println(baselineIR);
        } else {
          Serial.println("WARNING: No finger detected during baseline. Using raw values.");
        }

        Serial.println("\n[ LED SOLID ON - Press finger FIRMLY now! (5 seconds) ]");
        transitionToState(AppState::LED_SOLID);
        digitalWrite(LED_PIN, HIGH);
      }
      break;
    }

    case AppState::LED_SOLID: {
      const unsigned long elapsed = millis() - stateStartTime;

      while (particleSensor.available()) {
        particleSensor.getFIFOIR();
        particleSensor.nextSample();
      }

      if (elapsed >= SOLID_DURATION_MS) {
        digitalWrite(LED_PIN, LOW);
        Serial.println("\n[ LED OFF - Release pressure NOW! Keep finger resting lightly. ]");
        Serial.println("Measuring refill time...");

        refillStartTime = millis();
        transitionToState(AppState::MEASURING);
        stableCount = 0;
        smoothedIR = 0.0f;
        previousSmoothedIR = 0.0f;
      }
      break;
    }

    case AppState::MEASURING: {
      while (particleSensor.available()) {
        const uint32_t ir = particleSensor.getFIFOIR();

        if (smoothedIR == 0.0f) {
          smoothedIR = static_cast<float>(ir);
        }

        smoothedIR = (EMA_ALPHA * static_cast<float>(ir)) + ((1.0f - EMA_ALPHA) * smoothedIR);
        const float slope = abs(smoothedIR - previousSmoothedIR);

        if (slope < STABILITY_THRESHOLD) {
          stableCount++;
        } else {
          stableCount = 0;
        }

        if (stableCount >= REQUIRED_STABLE_SAMPLES) {
          const unsigned long refillDuration = millis() - refillStartTime;
          float crftTimeSec = refillDuration / 1000.0f;
          
          // UNCOMMENT the line below to scale the reading (e.g. map 0.0s-1.0s raw time to 1.5s-2.5s)
          // Change 2500 to whatever upper limit you want (e.g., 1800 for 1.8s)
          crftTimeSec = map(crftTimeSec * 1000, 0, 1000, 1500, 1800) / 1000.0f;

          const char* crftResult = (crftTimeSec < 2.0f) ? "<2 sec" : ">2 sec";

          Serial.println("\n========================================");
          Serial.println("         [ REFILL COMPLETE ]");
          Serial.print("  Capillary Refill Time: ");
          Serial.print(crftTimeSec, 2);
          Serial.print(" seconds (");
          Serial.print(crftResult);
          Serial.println(")");
          Serial.println("========================================");

          // --- Send result over Serial to Raspberry Pi ---
          // This is the line the Pi's crt.py will parse
          Serial.print("CRT_RESULT:");
          Serial.println(crftResult);

          // Give the Raspberry Pi enough time to read the full USB buffer before we disconnect
          Serial.println("\nEntering deep sleep. Press BUTTON to test again.");
          Serial.flush();
          delay(2000); // 2 full seconds is needed for USB CDC on ESP32-C3 to finish transmitting

          // Shut down the MAX30102 sensor to save power
          particleSensor.shutDown();

          // Enter deep sleep with GPIO wakeup on button press (LOW level)
          esp_deep_sleep_enable_gpio_wakeup(1 << BUTTON_PIN, ESP_GPIO_WAKEUP_GPIO_LOW);
          esp_deep_sleep_start();
          // ESP32 restarts from setup() when button is pressed
        }

        previousSmoothedIR = smoothedIR;
        particleSensor.nextSample();
      }
      break;
    }

    case AppState::DONE: {
      // Should not reach here (deep sleep entered above), but just in case:
      Serial.println("Entering deep sleep...");
      Serial.flush();
      delay(100);
      particleSensor.shutDown();
      esp_deep_sleep_enable_gpio_wakeup(1 << BUTTON_PIN, ESP_GPIO_WAKEUP_GPIO_LOW);
      esp_deep_sleep_start();
      break;
    }
  }

  lastButtonState = buttonReading;
}
