#include "tflmTimer.h"

uint8_t TflmTimer::ctr = 0;
measureMent *TflmTimer::measurements = nullptr;

void TflmTimer::measureTime(uint8_t identifier)
{
    uint32_t currentTime = ARM_CM_DWT_CYCCNT;
    measurements[ctr] = (measureMent){identifier, currentTime};
    ctr++;
}

void TflmTimer::startTimingService(Stream *serial)
{

    measurements = (measureMent *)malloc(sizeof(measureMent) * 150);
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    if (ARM_CM_DWT_CTRL != 0)
    {
        ARM_CM_DEMCR |= 1 << 24; // Set bit 24
        ARM_CM_DWT_CYCCNT = 0;
        ARM_CM_DWT_CTRL |= 1 << 0; // Set bit 0
    }
    else
    {
    }
}

void TflmTimer::getMeasurement_asByteArray(uint8_t* data) {
    uint32_t byteCtr  = 0;
    union convert converter;
    for (int i = 0; i < ctr; i++) {
        converter.data = measurements[i];
        for (int j = 0; j < sizeof(measureMent); j++) {
            data[byteCtr + j] = converter.buffer[j];
        }
        byteCtr += sizeof(measureMent);
    }
}

void TflmTimer::printMeasurements(Printer *printer)
{
    printer->println(TIMING, measurements, ctr);
}

void TflmTimer::clear() {
    ctr = 0;
}

uint32_t TflmTimer::getLenMeasureMents_inByte() {
    return ctr * sizeof(measureMent);
}