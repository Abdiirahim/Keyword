#include "features_generaator.h"

#include <cmath>
#include <cstring>

#include "features_model.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#define FIXED_POINT 16

namespace {

FrontendState g_micro_features_state;
bool g_is_first_time = true;

}  // namespace

TfLiteStatus InitializeMicroFeatures() {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;

  if (!FrontendPopulateState(&config, &g_micro_features_state, kAudioSampleFrequency)) {
    MicroPrintf("FrontendPopulateState() failed");
    return kTfLiteError;
  }

  g_is_first_time = true;
  return kTfLiteOk;
}

void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}

TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input = input;
  if (g_is_first_time) {
    g_is_first_time = false;
  }

  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_micro_features_state, frontend_input, input_size, num_samples_read);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;
    value -= 128;
    if (value < -128) value = -128;
    if (value > 127) value = 127;
    output[i] = value;
  }

  return kTfLiteOk;
}
