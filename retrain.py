import argparse
import os
from src.config import Config
parser = argparse.ArgumentParser(
                    prog='MicroNAS',
                    description='Sarches for NNs',
                    epilog='This is a nice piece of software')

parser.add_argument('--retrain_epochs', type=int, default=Config.retrain_epochs)
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--mcu', type=str) # NICLA or NUCLEO
# parser.add_argument('--num_retraining', type=int, default=5)
parser.add_argument('--path', type=str)
parser.add_argument('--full_train', type=bool, default=False)

from src.Utils.Experiment import Experiment


args = parser.parse_args()


EXP_PATH = args.path
MODEL_PATH = EXP_PATH + os.path.sep + "keras.h5"
experiment = Experiment.from_json(EXP_PATH)
print("RETRAIN_EXP")
print(experiment._dict)
experiment.clearEval()


Config.mcu = experiment._dict["config"]["mcu"]
Config.num_retraining = experiment._dict["config"]["num_retraining"]
Config.dataset = experiment._dict["config"]["dataset"]
Config.retrain_epochs = args.retrain_epochs

train_full = args.full_train

print("Train_full: ", train_full)

print("Config: ", Config.mcu, Config.num_retraining, Config.dataset)

import torch
from src.Nas.Networks.Pytorch.SearchNet import SearchNet
from src.Profiler.LatMemProfiler import set_ignore_latency, _lookUp
from src.Nas.Networks.Pytorch.SearchModule import InferenceType
from src.config import Config
import warnings
from src.Nas.Search import ArchSearcher
warnings.filterwarnings("ignore", category=FutureWarning)
from src.Utils.dataloader  import get_dataloaders, loadDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.TfLite.structure import TfLiteModel
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
torch.set_flush_denormal(True)

ucihar_train, ucihar_vali, ucihar_test = loadDataset(data_name=Config.dataset)
num_classes = ucihar_train.nb_classes
train_data_loader, vali_data_loader, test_data_loader, num_classes = get_dataloaders(ucihar_train, ucihar_vali, ucihar_test, num_classes, keras=False)
print("num_classes: ", num_classes)


data_shape = next(iter(train_data_loader))[0].shape
print("data_shape: ", data_shape)
ts_len = data_shape[1]
num_sensors = data_shape[2]




for _ in range(Config.num_retraining):
        



    import tensorflow as tf
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.layers import BatchNormalization
    import keras.backend as K
    import tensorflow_model_optimization as tfmot
    from tensorflow.keras.regularizers import l2


    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision



    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


    def getCallbacks():
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=Config.retrain_patience)
        return callback
    
    keras_model = load_model(MODEL_PATH, custom_objects = {"recall_m": recall_m, "precision_m": precision_m, "f1_m": f1_m})
    keras_model = keras_model.get_config()

    # Create a new model with the same architecture
    keras_model = keras.Model.from_config(keras_model)

    model_layers = keras_model.layers

    # Apply L2 regularization to specific layers
    for layer in model_layers:
        if isinstance(layer, keras.layers.Dense):  # Apply to Dense layers
            layer.kernel_regularizer = l2(0.01)  # Adjust the regularization strength as needed
        if isinstance(layer, keras.layers.Conv2D):  # Apply to Conv2D layers
            layer.kernel_regularizer = l2(0.01)  # Adjust the regularization strength as needed


    quantize_model = tfmot.quantization.keras.quantize_model
    keras_model = quantize_model(keras_model)


    ucihar_train_data_loader_keras, ucihar_vali_data_loader_keras, ucihar_test_data_loader_keras, uci_num_classes_keras = get_dataloaders(ucihar_train, ucihar_vali, ucihar_test, num_classes, keras=True)

    print(next(iter(ucihar_train_data_loader_keras))[0].shape)

    keras_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", Precision(), Recall(), f1_m])
    keras_model.build(input_shape=(1, 128, 9, 1))
    print(keras_model.summary())

    if not train_full:
        keras_model.fit(ucihar_train_data_loader_keras, epochs=Config.retrain_epochs, validation_data=ucihar_vali_data_loader_keras, callbacks=[getCallbacks()])
    else:
        keras_model.fit(ucihar_train_data_loader_keras, epochs=Config.retrain_epochs, validation_data=ucihar_vali_data_loader_keras)

    eval_test = keras_model.evaluate(ucihar_test_data_loader_keras)
    print("test_loss, test_acc, test_precision, test_recall:", eval_test)


    eval_val = keras_model.evaluate(ucihar_vali_data_loader_keras)
    print("val_loss, val_acc, val_precision, val_recall:", eval_val)


    keras_pred = keras_model.predict(ucihar_test_data_loader_keras)

    keras_pred = [np.argmax(x) for x in keras_pred]

    keras_true, keras_pred  = [], []

    for x_test, y_test in tqdm(ucihar_test_data_loader_keras):
        keras_pred.extend(np.argmax(keras_model.predict(x_test), axis=1))
        keras_true.extend(np.argmax(y_test, axis=1))


    keras_true_new = np.expand_dims(keras_true, 1)
    keras_pred_new = np.expand_dims(keras_pred, 1)
    m = MultiLabelBinarizer().fit(keras_true_new)


    acc_score = accuracy_score(keras_true_new, keras_pred_new)
    f1_weight = f1_score(m.transform(keras_true_new), m.transform(keras_pred_new), average="weighted")
    f1_samples = f1_score(m.transform(keras_true_new), m.transform(keras_pred_new), average="samples")
    f1_macro = f1_score(m.transform(keras_true_new), m.transform(keras_pred_new), average="macro")
    experiment.addEval(acc_score, -1, -1, f1_macro, full=train_full)
    print(acc_score)
    print(f1_weight)
    print(f1_macro)
    print(f1_samples)


    # Evaluate Quantized performance

    print(ucihar_train_data_loader_keras.__len__())
    tflm_model = TfLiteModel(keras_model, (ts_len, num_sensors, 1),  rep_dataset=ucihar_train_data_loader_keras)
    print("LEN_TFLM: ", len(tflm_model.byte_model))
    interpreter = tf.lite.Interpreter(model_content=tflm_model.byte_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    pred, true = [], []
    for x_test, y_test in tqdm(ucihar_test_data_loader_keras):
        for (x, y) in zip(x_test, y_test):
            input_scale, input_zero_point = input_details["quantization"]
            x_quant = (x / input_scale + input_zero_point).astype(np.int8)
            x_quant = np.expand_dims(x_quant, axis=0)
            interpreter.set_tensor(input_details['index'], x_quant)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            pred.append(output.argmax())
            true.append(np.argmax(y))

    acc_score = accuracy_score(true, pred)
    prec_score = precision_score(true, pred, average="macro")
    rec_score = recall_score(true, pred, average="macro")
    f1_sco = f1_score(true, pred, average="macro")
    print(acc_score)
    print(prec_score)
    print(rec_score)
    print(f1_sco)

    experiment.addEvalQuant(tflm_model, acc_score, -1, -1, f1_sco, full=train_full)