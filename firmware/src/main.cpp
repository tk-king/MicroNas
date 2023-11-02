#define CMSIS_NN
#include "Arduino.h"
#include "controls.h"
#include "helpers.h"
// #include "communicator.h"
#include "modelExecuter.h"
#include "tflmTimer.h"
#include "model_gen.h"
#include "printer.h"

// #define SERIAL_MODE

int modelLength = 0;
int inputLength = 0;

float dummyModelOutput[] = {1.5, 2.5, 3.5};

float *inputBuffer;

// Communicator comm;
Printer *printer;

void setup()
{
  delay(1000);
  printer = new Printer(&Serial);
#ifdef SERIAL_MODE
  comm.startUp(&Serial, printer);

#else
  Serial.begin(115200);
  while (!Serial)
  {
  }
  // Test code
  Serial.println("dummy_line_prevent_strange_characters");
  TflmTimer::startTimingService(&Serial);
  ModelExecuter *exec = new ModelExecuter(&Serial);
  exec->initTFLM();
  exec->setModel(model_gen, model_gen_len);
  float input[] = {3.14};
  float output[1];
  TflmTimer::measureTime(1);
  exec->execute();
  TflmTimer::measureTime(1);
  printer->println(INFERENCE_RESULT, output, 1);

  // print the time*/
  TflmTimer::printMeasurements(printer);
  TflmTimer::clear();
  // This needs to stay to indicate the end of the transmission
  Serial.println("---Complete---");
#endif
}

void loop()
{
}
