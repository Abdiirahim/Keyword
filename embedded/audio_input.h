#ifndef AUDIO_INPUT_HANDLER_H_
#define AUDIO_INPUT_HANDLER_H_

#include "tensorflow/lite/c/common.h"

// Get audio samples from mic buffer
TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples);

// Get the latest audio timestamp
int32_t LatestAudioTimestamp();

// Start microphone recording
TfLiteStatus InitAudioRecording();

#endif  // AUDIO_INPUT_HANDLER_H_
