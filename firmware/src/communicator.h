#ifndef communicator_H
#define communicator_H
#include "Arduino.h"
#include "controls.h"
#include "modelExecuter.h"
#include "tflmTimer.h"

#pragma pack(push, 1) 

// DATA_TYPES
const uint8_t TYPE_FLOAT = 1;


// General purpose struct for input / output arrays
struct commandReturn {
  uint8_t status; // == 0: No error
  uint8_t type; // Which data type
  uint32_t dataLen; // Len here is the number of bytes
  uint8_t *data; // The data in bytes
};

struct commandInput {
  uint8_t type; // The type of the data
  uint32_t dataLen; // The length of the data in bytes
  uint8_t *data; // The data in bytes
};


struct inputInference {
  uint8_t type; // The type of the data
  uint32_t inputLen; // The length of the data in bytes
  uint32_t outputLen; // The length of the output data 
  uint8_t *data; // The data in bytes
};

struct outputInference {
  uint8_t status; // == 0: No error
  uint8_t type; // The type of the data
  uint32_t dataLen; // The length of the neural network output
  uint32_t timeLen; // The length of the timing arrays
  uint32_t memoryConsumption; // the memory actually required by TFLM
  uint8_t *data; // The result of the Neural network execution
  uint8_t *timingMeasures; // The timing measures in array format <1 byte layerType, 4 byte cpuCycles>
};

struct command  {
  uint8_t cmd;
  uint32_t len;
};


#pragma pack(pop)
//int commandInputLenNoData = 5;

class Communicator {
  public:
    void startUp(Stream *serialObj, Printer *printer);
  private:
    Stream *serial;
    Printer *printer;
    uint8_t action;
    ModelExecuter *modelExec;
    int writeData(byte*, int len);
    void awaitCommand();
    void processInference();
    void sendOutStruct(commandReturn *outStruct);
    void sendOutStruct(outputInference *outStruct);
    void processModel();
    void processGetModel();
    void getModel();

};


#endif
