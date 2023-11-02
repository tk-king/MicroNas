#ifndef tflmTimer_H
#define tflmTimer_H

#include "printer.h"

#define ARM_CM_DEMCR (*(uint32_t *)0xE000EDFC)
#define ARM_CM_DWT_CTRL (*(uint32_t *)0xE0001000)
#define ARM_CM_DWT_CYCCNT (*(uint32_t *)0xE0001004)

// Timing vars
const uint8_t WHOLE_MODEL = 254;
const uint8_t TFLM_INVOKE = 255;

// CMSIS
const uint8_t CMSIS_ADD = 0;
const uint8_t CMSIS_CONV = 3;
const uint8_t CMSIS_DEPTHWISE_CONV = 4;
const uint8_t CMSIS_FULLY_CONNECTED = 9;
const uint8_t CMSIS_MUL = 18;
const uint8_t CMSIS_MAX_POOLING = 17;
const uint8_t CMSIS_AVG_POOLING = 1;
const uint8_t CMSIS_SOFTMAX = 25;
const uint8_t CMSIS_SVDF = 17;


// Generell Stuff
const uint8_t COMMMON_RELU = 19;
const uint8_t COMMMON_RELU6 = 21;
const uint8_t LOGISTIC = 14;
const uint8_t QUANTIZE = 114;
const uint8_t DEQUANTIZE = 6;
const uint8_t MEAN = 40;
const uint8_t EXPAND_DIMS = 70; 

union convert
{
    measureMent data;
    uint8_t buffer[sizeof(measureMent)];
};

class TflmTimer
{
public:
    static void measureTime(uint8_t identifier);
    static void startTimingService(Stream *serial);
    static void printMeasurements(Printer *printer);
    static void clear();
    static uint32_t getLenMeasureMents_inByte();
    static void getMeasurement_asByteArray(uint8_t *);

private:
    static uint8_t ctr;
    static measureMent *measurements;
};
#endif
