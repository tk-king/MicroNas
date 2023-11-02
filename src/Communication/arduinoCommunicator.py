import serial

FINISH_FLAG = "---Complete---"
ERROR_IDENTIFIER = "-- MbedOS Error Info --"

class ArduinoCommunicator():

    def __init__(self, port) -> None:
        self.serial = serial.Serial(port, baudrate=115200)
        self.serial.close()
        self.serial.open()
        self.serial.flushInput()
        self.serial.reset_input_buffer()
        
    def readLineString(self):
        data =  self.serial.readline()
        try:
            return data.decode("unicode_escape").strip()
        except Exception as e:
            print(data)
            raise e

    def read(self):
        return self.serial.readline().decode()

    def onComplete(self):
        self.serial.close()
