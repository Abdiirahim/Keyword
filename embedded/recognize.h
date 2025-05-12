#ifndef RECOGNIZE_H_
#define RECOGNIZE_H_

#include <cstdint>
#include "features_model.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"

class PreviousResultsQueue {
 public:
  PreviousResultsQueue() : front_index_(0), size_(0) {}

  struct Result {
    Result() : time_(0), scores() {}
    Result(int32_t time, int8_t* input_scores) : time_(time) {
      for (int i = 0; i < kCategoryCount; ++i) {
        scores[i] = input_scores[i];
      }
    }
    int32_t time_;
    int8_t scores[kCategoryCount];
  };

  int size() { return size_; }
  bool empty() { return size_ == 0; }
  Result& front() { return results_[front_index_]; }
  Result& back() {
    int back_index = front_index_ + (size_ - 1);
    if (back_index >= kMaxResults) back_index -= kMaxResults;
    return results_[back_index];
  }

  void push_back(const Result& entry) {
    if (size_ >= kMaxResults) {
      MicroPrintf("Too many results!");
      return;
    }
    size_ += 1;
    back() = entry;
  }

  Result pop_front() {
    if (size_ <= 0) {
      MicroPrintf("No results to remove!");
      return Result();
    }
    Result result = front();
    front_index_ += 1;
    if (front_index_ >= kMaxResults) front_index_ = 0;
    size_ -= 1;
    return result;
  }

  Result& from_front(int offset) {
    if ((offset < 0) || (offset >= size_)) {
      MicroPrintf("Index out of range!");
      offset = size_ - 1;
    }
    int index = front_index_ + offset;
    if (index >= kMaxResults) index -= kMaxResults;
    return results_[index];
  }

 private:
  static constexpr int kMaxResults = 50;
  Result results_[kMaxResults];
  int front_index_;
  int size_;
};

class RecognizeCommands {
 public:
  explicit RecognizeCommands(int32_t average_window_duration_ms = 1000,
                             uint8_t detection_threshold = 200,
                             int32_t suppression_ms = 1500,
                             int32_t minimum_count = 3);

  TfLiteStatus ProcessLatestResults(const TfLiteTensor* latest_results,
                                    const int32_t current_time_ms,
                                    const char** found_command, uint8_t* score,
                                    bool* is_new_command);

 private:
  int32_t average_window_duration_ms_;
  uint8_t detection_threshold_;
  int32_t suppression_ms_;
  int32_t minimum_count_;
  PreviousResultsQueue previous_results_;
  const char* previous_top_label_;
  int32_t previous_top_label_time_;
};

#endif  // RECOGNIZE_H_
