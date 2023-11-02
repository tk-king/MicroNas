#include "modelExecuter.h"
#include "Arduino.h"

ModelExecuter::ModelExecuter(Stream *serial)
{
  this->serial = new Printer(serial);
  // Keep size fixed for now
  this->tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
  if (this->tensor_arena == NULL)
  {
    Serial.println("Could not get enough memory");
  }
}

void ModelExecuter::initTFLM()
{
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  if (resolver.AddConv2D() != kTfLiteOk) {
    Serial.println(F("Error allocating Conv2D"));
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println(F("Error allocating SoftMax"));
  }
  if (resolver.AddRelu() != kTfLiteOk) {
    Serial.println(F("Error allocating Relu"));
  }
  if (resolver.AddMean() != kTfLiteOk) {
    Serial.println(F("Error allocating Mean"));
  }
  if (resolver.AddAdd() != kTfLiteOk) {
    Serial.println(F("Error allocating Add"));
  }
}

void ModelExecuter::setModel(const unsigned char *newModel, uint32_t modelLength)
{
  //free((void *)newModel);
  //this->model = newModel;
  this->modelLength = modelLength;
  // Set model here and build interpreter
  this->tflmModel = tflite::GetModel(newModel);
  if (this->tflmModel->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         this->tflmModel->version(), TFLITE_SCHEMA_VERSION);
  }
  static tflite::RecordingMicroInterpreter static_interpreter(
      this->tflmModel, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter == NULL)
  {
    serial->printError("Interpreter is null");
  }
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    serial->printError("Allocation of tensors failed");
    return;
  }
  // Grab the heap statistics
  // mbed_stats_heap_t heap_stats;
  // mbed_stats_heap_get(&heap_stats);

  // Log memory stuff
  // serial->println(MEMORY, "HEAP_CURRENT_SIZE", heap_stats.current_size);
  // serial->println(MEMORY, "HEAP_RESERVED_SIZE", heap_stats.reserved_size);
  serial->println(MEMORY, "kTensorAreaUsed", interpreter->arena_used_bytes());
  interpreter->GetMicroAllocator().PrintAllocations();

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
  serial->printStatus("Got Model");
}

const unsigned char *ModelExecuter::getModel()
{
  return this->model;
}

uint32_t ModelExecuter::getModelLength()
{
  return this->modelLength;
}

uint32_t ModelExecuter::getUsedMemory()
{
  return interpreter->arena_used_bytes();
}

int ModelExecuter::execute()
{

  noInterrupts();
  if (kTfLiteOk != interpreter->Invoke())
  {
    interrupts();
    return -1;
  }
  interpreter->GetMicroAllocator().PrintAllocations();
  interrupts();
  return 0;
}