#ifndef modelExecuter_H
#define modelExecuter_H

#include "Arduino.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_profiler.h"

// #include "mbed.h"
// #include "mbed_mem_trace.h"
#include "printer.h"


// static tflite::AllOpsResolver resolver;
static tflite::MicroMutableOpResolver<5> resolver;

class ModelExecuter {

  public:
    ModelExecuter(Stream *serial);
    void setModel(const unsigned char *model, uint32_t modelLength);
    const unsigned char* getModel();
    uint32_t getModelLength();
    int execute();
    void initTFLM();
    uint32_t getUsedMemory();
  private:
    const unsigned char *model;
    uint32_t modelLength;
    Printer *serial;

    // TFLM stuff
    tflite::ErrorReporter* error_reporter = nullptr;
    tflite::RecordingMicroInterpreter* interpreter = nullptr;
    //tflite::RecordingMicroInterpreter* interpreter = nullptr;
    const tflite::Model* tflmModel = nullptr;
    TfLiteTensor* inputTensor = nullptr;
    TfLiteTensor* outputTensor = nullptr;
    uint8_t *tensor_arena;
    #if defined(ARDUINO_NICLA)
      int kTensorArenaSize = 1024*38; // Nicla
    #elif defined(ARDUINO_NUCLEO_F446RE)
      int kTensorArenaSize = 1024 * 120; 
    #else
      int kTensorArenaSize = 1024 * 150; // Nucelo
    #endif
};

#endif
