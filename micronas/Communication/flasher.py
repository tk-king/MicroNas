from enum import Enum
import os
from micronas.Utilities import RecorderError, tflmBackend
from micronas.Communication.arduinoCommunicator import ArduinoCommunicator, ERROR_IDENTIFIER, FINISH_FLAG
from micronas.config import Config, MicroNasMCU

def readOutput(port=Config.port):
    lines = []
    comm = ArduinoCommunicator(port)
    line = comm.readLineString()
    error = RecorderError.NOERROR
    errorText = ""
    ctr = 0
    while line != FINISH_FLAG and line != ERROR_IDENTIFIER :
        lines.append(line)
        line = comm.serial.readline()
        try:
            line = line.decode("utf-8").strip()
            if str(line).startswith("Error"):
                error = RecorderError.RUNTIMEERROR
                errorText = line
        except:
            pass
        ctr += 1
    comm.onComplete()
    error = RecorderError.NOERROR
    if (line == ERROR_IDENTIFIER):
        error = RecorderError.RUNTIMEERROR
        errorText = "System crash"
        return lines, error, errorText
    for line in lines:
        if str(line).startswith("Error"):
            return lines, RecorderError.RUNTIMEERROR, line
    return lines, error, errorText

def flashNicla(firmwarePath):
    import subprocess as sp
    import os
    flashCommnad = f"pio run -t upload --project-dir firmware"
    # stdout=sp.Pipe for output
    child = sp.Popen(flashCommnad, stdout=None, shell=True)
    output = child.communicate()[0]
    rc = child.returncode
    return output, RecorderError.FLASHERROR if rc else RecorderError.NOERROR

def getConfig(mcu: MicroNasMCU, lib: tflmBackend):

    mcu = MicroNasMCU(mcu)
    lib = tflmBackend(lib)
    
    envs = {MicroNasMCU.NiclaSense: "[env:nicla_sense_me]", MicroNasMCU.NUCLEOL552ZEQ: "[env:nucleo_l552ze_q]", MicroNasMCU.NUCLEOF446RE: "[env:nucleo_f446re]"}
    platforms = {MicroNasMCU.NiclaSense: "nordicnrf52@9.4.0", MicroNasMCU.NUCLEOL552ZEQ: "ststm32", MicroNasMCU.NUCLEOF446RE: "ststm32"}
    boards = {MicroNasMCU.NiclaSense: "nicla_sense_me", MicroNasMCU.NUCLEOL552ZEQ: "nucleo_l552ze_q", MicroNasMCU.NUCLEOF446RE: "nucleo_f446re"}

    libs = {tflmBackend.STOCK: "tflite_micro", tflmBackend.CMSIS: "tflite_micro_stock"}

    framework = "arduino"
    monitor_speed = "115200"

    return f"{envs[mcu]}\nplatform = {platforms[mcu]}\nboard = {boards[mcu]}\nframework = {framework}\nmonitor_speed={monitor_speed}\nlib_ignore = {libs[lib]}"



def configureFirmware(byteModel, firmwarePath, mcu: MicroNasMCU, tflmB: tflmBackend):
    byteString = ','.join([hex(x) for x in byteModel])
    if not os.path.exists(firmwarePath):
        open(firmwarePath, 'w').close()

    # Write the model
    with open(os.path.join(firmwarePath, "src/model_gen.h"), 'w') as f:
        f.write(f'alignas(8) const unsigned char PROGMEM model_gen[] = {{ \n {byteString} \n }};\n')
        f.write(f'unsigned int model_gen_len = {len(byteModel)};')
    
    # # Decide which tensorflow implementation to
    # with open(os.path.join(firmwarePath, "platformio.ini"), 'r') as f_read:
    #     allLines = f_read.readlines()

    with open(os.path.join(firmwarePath, "platformio.ini"), "w") as f:
        f.write(getConfig(mcu, tflmB))

    # with open(os.path.join(firmwarePath, "platformio.ini"), "w") as f:
    #     resLines = []
    #     for line in allLines:
    #         if "lib_ignore" in line:
    #             if tflmB == tflmBackend.STOCK: 
    #                 resLines.append("lib_ignore = tflite_micro")
    #             elif tflmB == tflmBackend.CMSIS:
    #                 resLines.append("lib_ignore = tflite_micro_stock")
    #         else:
    #             resLines.append(line)
    #     f.write("".join(resLines))  