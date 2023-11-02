import numpy as np
import random
import torch
import tensorflow as tf
import tensorflow_model_optimization as tfmot

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)

import argparse
from src.config import Config
parser = argparse.ArgumentParser(
                    prog='MicroNAS',
                    description='Sarches for NNs',
                    epilog='This is a nice piece of software')

parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--target_latency', type=int)
parser.add_argument('--target_memory', type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--mcu', type=str) # NICLA or NUCLEO
parser.add_argument('--num_retraining', type=int, default=5)
parser.add_argument('--eps', type=float)

args = parser.parse_args()
Config.mcu = args.mcu
Config.search_epochs = args.epochs
Config.target_lat = args.target_latency
Config.target_mem = args.target_memory
Config.num_retraining = args.num_retraining
Config.dataset = args.dataset
Config.eps = args.eps


from src.Utils.Experiment import Experiment
import torch
from src.Nas.Networks.Pytorch.SearchNet import SearchNet
from src.Profiler.LatMemProfiler import set_ignore_latency, _lookUp
from src.Nas.Networks.Pytorch.SearchModule import InferenceType
from src.config import Config
import torch
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
torch.set_flush_denormal(True)
import tensorflow.keras as keras

class_weights = None
if Config.dataset == "skodar":
    class_weights = [0.8583, 0.7315, 0.77, 1.0527, 1.7239, 1.9238, 0.9947, 0.8907, 0.7788, 1.4141]

if Config.dataset == "ucihar":
    class_weights = [1.0052, 1.1525, 1.2436, 0.951, 0.8872, 0.8661]

if class_weights == None:
    raise Exception("Class weights not known")

experiment = Experiment()

datasetName = args.dataset
ucihar_train, ucihar_vali, ucihar_test = loadDataset(data_name=datasetName)
num_classes = ucihar_train.nb_classes
train_data_loader, vali_data_loader, test_data_loader, num_classes = get_dataloaders(ucihar_train, ucihar_vali, ucihar_test, num_classes, keras=False)

print(type(train_data_loader))

print("num_classes: ", num_classes)


data_shape = next(iter(train_data_loader))[0].shape
print("data_shape: ", data_shape)
ts_len = data_shape[1]
num_sensors = data_shape[2]

# ts_len = 128
# num_sensors = 9

print("Num_lookup: ", len(_lookUp))

set_ignore_latency(False)

nas_net = SearchNet([ts_len, num_sensors], num_classes).to(Config.device)
weights = nas_net.get_nas_weights()
fake_input = torch.randn((1, 1, ts_len, num_sensors)).to(Config.device)
print("output_shape: ", nas_net(fake_input))
searcher = ArchSearcher(nas_net)


print("Eps: ", nas_net._t)
target_lat = args.target_latency
target_mem = args.target_memory
epochs = args.epochs
epochs_pretrain = 0
searcher.train(train_data_loader, vali_data_loader, epochs, alpha_lat=0.05, alpha_mem=0.2, target_lat=target_lat, target_mem=target_mem, epochs_pretrain=epochs_pretrain, eps_decay=Config.eps)



# nas_net.print_nas_weights(nas_net._t)
fake_input_128 = torch.randn((1, 1, ts_len, num_sensors))
print(fake_input_128.shape)
# print(nas_net._t)
# old_t = nas_net._t
# nas_net._t = 1e-9
hw_metrics = nas_net(fake_input_128, inf_type=InferenceType.MAX_WEIGHT)[1:]
print(hw_metrics)
experiment.addHardwarePrediction(hw_metrics[0], hw_metrics[1])





for _ in range(Config.num_retraining):
        
    keras_model = nas_net.getKeras(getPruned=True, batch_size=None, inf_type=InferenceType.MAX_WEIGHT)
    keras_model = keras_model.get_config()
    keras_model = keras.Model.from_config(keras_model)
    keras_model.save("models/search_keras_model.h5")
    experiment.addKerasModel(keras_model)
    quantize_model = tfmot.quantization.keras.quantize_model
    keras_model = quantize_model(keras_model)


    import tensorflow as tf
    from tensorflow.keras.metrics import Precision, Recall
    import keras.backend as K
    # import tensorflow_model_optimization as tfmot


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


    ucihar_train_data_loader_keras, ucihar_vali_data_loader_keras, ucihar_test_data_loader_keras, uci_num_classes_keras = get_dataloaders(ucihar_train, ucihar_vali, ucihar_test, num_classes, keras=True)

    print(next(iter(ucihar_train_data_loader_keras))[0].shape)

    keras_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", Precision(), Recall(), f1_m])
    keras_model.build(input_shape=(1, 128, 9, 1))
    print(keras_model.summary())

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
    experiment.addEval(acc_score, -1, -1, f1_macro)
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

    experiment.addEvalQuant(tflm_model, acc_score, -1, -1, f1_sco)