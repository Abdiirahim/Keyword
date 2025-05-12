#include <TensorFlowLite.h>

#include "audio_input.h"
#include "command_responder.h"
#include "feature_hander.h"
#include "main_functions.h"
#include "features_model.h"
#include "model.h"
#include "recognize.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#undef PROFILE_MICRO_SPEECH

// Global variables
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 10 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}

// Runs once at the beginning
void setup() {
  tflite::InitializeTarget();

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model version mismatch.");
    return;
  }

  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) return;
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return;
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) return;
  if (micro_op_resolver.AddReshape() != kTfLiteOk) return;

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Tensor allocation failed.");
    return;
  }

  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    MicroPrintf("Model input format error.");
    return;
  }

  model_input_buffer = model_input->data.int8;

  static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;

  if (InitAudioRecording() != kTfLiteOk) {
    MicroPrintf("Audio init failed.");
    return;
  }

  MicroPrintf("Setup complete.");
}

// Repeats continuously
void loop() {
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;

  if (feature_provider->PopulateFeatureData(previous_time, current_time, &how_many_new_slices) != kTfLiteOk) {
    MicroPrintf("Feature creation failed.");
    return;
  }

  previous_time += how_many_new_slices * kFeatureSliceStrideMs;
  if (how_many_new_slices == 0) return;

  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Model failed to run.");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);

  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;

  if (recognizer->ProcessLatestResults(output, current_time, &found_command, &score, &is_new_command) != kTfLiteOk) {
    MicroPrintf("Recognition failed.");
    return;
  }

  RespondToCommand(current_time, found_command, score, is_new_command);
}
