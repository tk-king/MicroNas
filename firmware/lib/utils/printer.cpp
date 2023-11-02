#include "printer.h"

Printer::Printer(Stream *stream)
{
    this->serial = stream;
}

void Printer::print(const char *command, const char *message)
{
    serial->print(command);
    serial->print(" ");
    serial->println(message);
}

void Printer::println(const char *command, const char *message)
{
    this->print(command, message);
    serial->println("");
}

void Printer::println(const char *command, float *data, int dataLen)
{
    serial->print(command);
    serial->print(" ");
    for (int i = 0; i < dataLen; i++)
    {
        serial->print(data[i]);
        if (i < (dataLen - 1))
        {
            serial->print(", ");
        }
    }
    serial->println("");
}

void Printer::println(const char* command, int firstValue, int secondValue) {
    serial->print(command);
    serial->print(" (");
    serial->print(firstValue);
    serial->print(", ");
    serial->print(secondValue);
    serial->println(")");
}

void Printer::printStatus(const char* message) {
    serial->print("Status ");
    serial->println(message);
}

void Printer::printError(const char* message) {
    serial->print("Error ");
    serial->println(message);
}

void Printer::println(const char* command, const char* identifier, int secondValue) {
    serial->print(command);
    serial->print(" (");
    serial->print(identifier);
    serial->print(",");
    serial->print(secondValue);
    serial->println(")");
}

void Printer::println(const char *command, measureMent *data, int dataLen)
{
    serial->print(command);
    serial->print(" ");
    for (int i = 0; i < dataLen; i++)
    {
        {
            serial->print("(");
            serial->print(data[i].layerType);
            serial->print(",");
            serial->print(data[i].time);
            serial->print(")");
            if (i < (dataLen - 1))
            {
                serial->print(",");
            }
        }
    }
    serial->println("");
}