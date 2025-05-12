#ifndef FEATURES_GENERATOR_H_
#define FEATURES_GENERATOR_H_

#include "tensorflow/lite/c/common.h"

// Set up the feature generation system
TfLiteStatus InitializeMicroFeatures();

// Generate features from audio input
TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read);

#endif  // FEATURES_GENERATOR_H_
