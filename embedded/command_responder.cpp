#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include "command_responder.h"   
#include "tensorflow/lite/micro/micro_log.h"

// Reacts to voice commands by lighting up LEDs
void RespondToCommand(int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    is_initialized = true;
  }

  static int32_t last_command_time = 0;
  static int count = 0;

  if (is_new_command) {
    MicroPrintf("Heard %s (%d) @%dms", found_command, score, current_time);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    if (found_command[0] == 'g') {
      digitalWrite(LEDG, LOW);  // Green for "guess"
    } else if (found_command[0] == 'n') {
      digitalWrite(LEDR, LOW);  // Red for "no"
    } else if (found_command[0] == 'u') {
      digitalWrite(LEDB, LOW);  // Blue for unknown
    }

    last_command_time = current_time;
  }

  // Turn off LEDs after 3 seconds
  if (last_command_time != 0 && last_command_time < (current_time - 3000)) {
    last_command_time = 0;
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
  }

  // Blink built-in LED every inference
  ++count;
  digitalWrite(LED_BUILTIN, count & 1 ? HIGH : LOW);
}

#endif  // ARDUINO_EXCLUDE_CODE
