#ifndef controls_H
#define controls_H

// Structs for sending stuff
#pragma pack(push, 1)
typedef struct
{
    uint8_t layerType;
    uint64_t time;
} measureMent;
#pragma pack(pop)

// Commands
const uint8_t SEND_MODEL = 1;
const uint8_t GET_MODEL = 2;
const uint8_t INFERENCE = 3;

// DataTypes
const uint8_t DATA_NO_DATA = 0;
const uint8_t DATA_FLOAT = 1;
const uint8_t DATA_BYTE = 2;

// Status codes
const uint8_t COMMAND_NOT_FOUND = 1;


// Status codes for flash mode
const char INFERENCE_RESULT[] = "INF_RESULT";
const char TIMING[] = "TIMING";
const char MEMORY[] = "MEMORY"; 
const char CUSTOM_ERROR[] = "ERROR";

#endif
