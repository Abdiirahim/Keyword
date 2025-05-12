#ifndef FEATURES_MODEL_H_
#define FEATURES_MODEL_H_

// Audio settings
constexpr int kMaxAudioSampleSize = 512;
constexpr int kAudioSampleFrequency = 16000;

// Feature settings
constexpr int kFeatureSliceSize = 40;
constexpr int kFeatureSliceCount = 49;
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 20;
constexpr int kFeatureSliceDurationMs = 30;

// Output labels
constexpr int kSilenceIndex = 0;
constexpr int kUnknownIndex = 1;
constexpr int kCategoryCount = 4;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // FEATURES_MODEL_H_
