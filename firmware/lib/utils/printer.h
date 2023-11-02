#ifndef PRINTER_H
#define PRINTER_H

#include "Arduino.h"
#include "controls.h"


class Printer {
    public:
        Printer(Stream * stream);
        void println(const char* command, const char* message);
        void print(const char* command, const char* message);
        void println(const char* command, float* data, int dataLen);
        void println(const char* command, measureMent *data, int dataLen);
        void println(const char* command, int firstValue, int secondValue);
        void println(const char* command, const char* identifier, int secondValue);
        void printError(const char* message);
        void printStatus(const char* message);
    private:
        Stream *serial;
};


#endif