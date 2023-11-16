
import tensorflow as tf
import numpy as np

def exec_tflm(dataloaderKeras, tflmModel):
    interpreter = tf.lite.Interpreter(model_content=tflmModel)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    pred, true = [], []
    for x_test, y_test in dataloaderKeras:
        for (x, y) in zip(x_test, y_test):
            input_scale, input_zero_point = input_details["quantization"]
            x_quant = (x / input_scale + input_zero_point).astype(np.int8)
            x_quant = np.expand_dims(x_quant, axis=0)
            x_quant = np.expand_dims(x_quant, axis=-1)
            interpreter.set_tensor(input_details['index'], x_quant)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            pred.append(output.argmax())
            true.append(np.argmax(y))
    return pred