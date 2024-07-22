from micronas.Utils.PytorchKerasAdapter import PytorchKerasAdapter
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf

def eval_keras(model, num_classes, train_dataloader, vali_dataloader, test_dataloader):

    res = {}

    loaders = [train_dataloader, vali_dataloader, test_dataloader]
    names = ["train", "vali", "test"]
    for dataloader, data_name in zip(loaders, names):
        dataloader_keras = PytorchKerasAdapter(dataloader, num_classes)
        
        keras_pred = []
        keras_true = []
        for x_test, y_test in tqdm(dataloader_keras):
            keras_pred.extend(np.argmax(model.predict(x_test, verbose=0), axis=1))
            keras_true.extend(np.argmax(y_test, axis=1))
            
        keras_true_new = np.expand_dims(keras_true, 1)
        keras_pred_new = np.expand_dims(keras_pred, 1)
        m = MultiLabelBinarizer().fit(keras_true_new)

    
        acc = accuracy_score(keras_true_new, keras_pred_new)
        f1 = f1_score(keras_true_new, keras_pred_new, average="macro")
        res[f"acc_{data_name}"] = acc
        res[f"f1_{data_name}"] = f1
    return res

def eval_keras_quant(tf_lite_model, num_classes, train_dataloader, vali_dataloader, test_dataloader):
    
    res = {}
    loaders = [train_dataloader, vali_dataloader, test_dataloader]
    names = ["train", "vali", "test"]

    for dataloader, data_name in zip(loaders, names):
        interpreter = tf.lite.Interpreter(model_content=tf_lite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        dataloader_keras = PytorchKerasAdapter(dataloader, num_classes)
        pred, true = [], []
        for x_test, y_test in tqdm(dataloader_keras):
            for (x, y) in zip(x_test, y_test):
                input_scale, input_zero_point = input_details["quantization"]
                x_quant = (x / input_scale + input_zero_point).astype(np.int8)
                x_quant = np.expand_dims(x_quant, axis=0)
                interpreter.set_tensor(input_details['index'], x_quant)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details["index"])[0]
                pred.append(output.argmax())
                true.append(np.argmax(y))
    
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="macro")
        res[f"acc_quant_{data_name}"] = acc
        res[f"f1_quant_{data_name}"] = f1
    return res