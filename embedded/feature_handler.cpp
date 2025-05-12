#include "feature_hander.h"

#include "audio_input.h"
#include "features_generaator.h"
#include "features_model.h"
#include "tensorflow/lite/micro/micro_log.h"

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(int32_t last_time_in_ms,
                                                  int32_t time_in_ms,
                                                  int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    MicroPrintf("Feature size mismatch %d vs %d", feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
  int slices_needed = ((((time_in_ms - last_time_in_ms) - kFeatureSliceDurationMs) *
                        kFeatureSliceStrideMs) /
                           kFeatureSliceStrideMs +
                       kFeatureSliceStrideMs) /
                      kFeatureSliceStrideMs;

  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures();
    if (init_status != kTfLiteOk) return init_status;
    is_first_run_ = false;
    return kTfLiteOk;
  }

  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  if (slices_needed == 0) return kTfLiteOk;

  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;

  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest = feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src = feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest[i] = src[i];
      }
    }
  }

  for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount; ++new_slice) {
    const int new_step = last_step + (new_slice - slices_to_keep);
    const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
    int16_t* audio_samples = nullptr;
    int audio_samples_size = 0;
    GetAudioSamples(slice_start_ms, kFeatureSliceDurationMs,
                    &audio_samples_size, &audio_samples);
    constexpr int wanted =
        kFeatureSliceDurationMs * (kAudioSampleFrequency / 1000);
    if (audio_samples_size != wanted) {
      MicroPrintf("Audio size %d too small, want %d", audio_samples_size, wanted);
      return kTfLiteError;
    }
    int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
    size_t num_samples_read;
    TfLiteStatus status = GenerateMicroFeatures(
        audio_samples, audio_samples_size, kFeatureSliceSize, new_slice_data,
        &num_samples_read);
    if (status != kTfLiteOk) return status;
  }

  return kTfLiteOk;
}
