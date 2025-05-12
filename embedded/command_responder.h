#ifndef AUDIO_COMMAND_RESPONDER_H_
#define AUDIO_COMMAND_RESPONDER_H_

#include "tensorflow/lite/c/common.h"

// Runs when a command is detected.
void RespondToCommand(int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command);

#endif  // AUDIO_COMMAND_RESPONDER_H_
