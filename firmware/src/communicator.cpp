#include "communicator.h"

void Communicator::startUp(Stream *serialObj, Printer *printerObj)
{
  serial = serialObj;
  printer = printerObj;
  modelExec = new ModelExecuter(serialObj);

  while (!serial->available())
    ;
  TflmTimer::startTimingService(serialObj);
  while (true)
  {
    awaitCommand();
  }
}

int Communicator::writeData(byte *data, int len)
{
  while (!serial->available())
    ;
  return serial->write(data, len);
}

void Communicator::awaitCommand()
{
  while (serial->available())
  {
    action = serial->read();
    switch (action)
    {
    case (INFERENCE):
      processInference();
      break;
    case (SEND_MODEL):
      processModel();
    case (GET_MODEL):
      getModel();
    default:
      serial->write(COMMAND_NOT_FOUND);
      break;
    }
  }
}

void Communicator::sendOutStruct(commandReturn *outStruct)
{
  serial->write((uint8_t *)outStruct, 6);
  serial->write(outStruct->data, outStruct->dataLen);
}

void Communicator::sendOutStruct(outputInference *outStruct)
{
  serial->write((u_int8_t *)outStruct, 14);
  serial->write(outStruct->data, outStruct->dataLen);
  serial->flush();
  serial->write(outStruct->timingMeasures, outStruct->timeLen);
  serial->flush();
}

/*void Communicator::processInference() {
  commandInput input = commandInput();
  serial->readBytes((uint8_t*) &input, 5);
  int numFloats = input.dataLen / sizeof(float);
  float arr[numFloats];
  input.data = (byte*) &arr;
  serial->readBytes(input.data, input.dataLen);

  // TODO: Replace this logic with actual inference on the model
  float outArr[input.dataLen];
  for (int i = 0; i < input.dataLen; i++) {
    outArr[i] = arr[i] + 1;
  }
  
  commandReturn retStruct = commandReturn();
  retStruct.status = 0;
  retStruct.type = 1;
  retStruct.dataLen = input.dataLen;
  retStruct.data = (byte*) outArr; 
  sendOutStruct(&retStruct);
}*/

void Communicator::processModel()
{
  uint32_t modelLen;
  serial->readBytes((uint8_t *) &modelLen, 4);
  uint8_t *model = (uint8_t *)malloc(modelLen);

  serial->readBytes(model, modelLen);
  serial->print("ModelLen ");
  serial->println(modelLen);
  // modelExec->setModel(model, modelLen);

  // TflmTimer::measureTime(1);
  // modelExec->execute();
  // TflmTimer::measureTime(1);

  // // print the time*/
  // TflmTimer::printMeasurements(printer);
  // This needs to stay to indicate the end of the transmission
  Serial.println("---Complete---");
}

void Communicator::processInference()
{
  inputInference input = inputInference();
  serial->readBytes((uint8_t *)&input, 9);
  int numFloats = input.inputLen / sizeof(float);
  float inputArr[numFloats];
  input.data = (uint8_t *)&inputArr;
  float outputArr[input.outputLen];
  serial->readBytes(input.data, input.inputLen);
  outputInference output = outputInference();
  output.data = (byte *)outputArr;
  output.dataLen = input.outputLen;
  output.type = TYPE_FLOAT;

  TflmTimer::clear();

  modelExec->execute();

  output.timeLen = TflmTimer::getLenMeasureMents_inByte();
  output.memoryConsumption = modelExec->getUsedMemory();
  uint8_t timingMeasureMents[output.timeLen];
  TflmTimer::getMeasurement_asByteArray(timingMeasureMents);
  output.timingMeasures = timingMeasureMents;
  sendOutStruct(&output);
  TflmTimer::clear();
}

void Communicator::getModel()
{
  commandInput input = commandInput();
  serial->readBytes((uint8_t *)&input, 5);
  commandReturn ret = commandReturn();
  ret.status = 0;
  ret.type = DATA_BYTE;
  ret.dataLen = modelExec->getModelLength();
  ret.data = (uint8_t *)modelExec->getModel();
  sendOutStruct(&ret);
}
