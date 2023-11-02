from dataclasses import dataclass
from src.config import Config
import json
from datetime import datetime
from pathlib import Path
import os
import re

def extract_date_time_from_filename(filename):
    pattern = r"_(\d{2}_\d{2}_\d{4}\d{2}_\d{2}_\d{2})"  # Regular expression pattern to match the date and time
    match = re.search(pattern, filename)
    
    if match:
        date_time_string = match.group(1)
        return date_time_string
    else:
        return None

class Experiment():
    def __init__(self):
        super().__init__()
    
        self._dict = {"eval": [], "eval_quant": []}
        self._keras_model = None
        self._tflmModel = []
        now = datetime.now()
        self._dt_string = now.strftime("%d_%m_%Y%H_%M_%S")
        self.addConfig()

    @classmethod
    def from_json(cls, exp_path):
        with open(exp_path + os.path.sep + "record.json", "r") as f:
            experiment_data = json.load(f)
        # Extract data from JSON and initialize Experiment instance
        experiment = cls()
        experiment._dict = experiment_data
        print("JSON data")
        print(experiment._dict)
        experiment._dt_string = extract_date_time_from_filename(exp_path)
        return experiment

    def addSearchLogger(self, logger):
        self._dict["searchLogger"] = {}
        self._dict["searchLogger"]["loss_out"] = logger._loss_out
        self._dict["searchLogger"]["loss_lat"] = logger._loss_lat
        self._dict["searchLogger"]["loss_mem"] = logger._loss_mem
        self._dict["searchLogger"]["latency"] = logger._loss_mem
        self._dict["searchLogger"]["memory"] = logger._memory
        self.write()

    def addConfig(self):
        cfg = dict(vars(type(Config)))
        cfg.update(vars(Config))
        newCfg = {}
        for k, v in cfg.items():
            if "__" not in k:
                newCfg[k] = v
        self._dict["config"] = newCfg
        print("config: ", self._dict["config"])
        self.write()

    def clearEval(self):
        self._dict["eval_full"] = []
        self._dict["eval_quant_full"] = []

    def addEvalQuant(self, tfLiteModel, acc_score, prec_store_macro, rec_score_macro, f1_score_macro, full=False):
        data = {
            "accuracy": acc_score,
            "precision": prec_store_macro,
            "recall": rec_score_macro,
            "f1_macro": f1_score_macro
        }

        key = "eval_quant_full" if full else "eval_quant"
        if key not in self._dict:
            self._dict[key] = []
        self._dict[key].append(data)

        self._dict["eval_quant"].append(data)
        self._tflmModel.append(tfLiteModel)
        self.write()

    def addEval(self, acc_score, prec_store_macro, rec_score_macro, f1_score_macro, full=False):
        data = {
            "accuracy": acc_score,
            "precision": prec_store_macro,
            "recall": rec_score_macro,
            "f1_macro": f1_score_macro
        }

        key = "eval_full" if full else "eval"
        if key not in self._dict:
            self._dict[key] = []
        self._dict[key].append(data)
        self.write()

    # def addQuantizedEval(self, acc_score, prec_score, rec_score, f1_score):
    #     self._dict["eval_quant"] = [acc_score, prec_score, rec_score, f1_score]

    def addActualTiming(self, timing):
        self._dict["actual_timing"] = timing
        self.write()

    
    def addKerasModel(self, model):
        self._keras_model = model
        self.write()

    def addHardwarePrediction(self, latency, memory):
        self._dict["pred_latency"] = latency.item()
        self._dict["pred_memory"] = memory.item()
        self.write()

    def addNasWeights(self, weights):
        self._dict["nas_weights"] = weights
        self.write()

    # def addEval(self, eval_test, eval_val):
    #     self._dict["eval_test"] = eval_test
    #     self._dict["eval_val"] = eval_val

    def print(self):
        print(self._dict)

    def add(self, key, value):
        self._dict[key] = value
        self.write()

    def addTFLMModel(self, tflmModel):
        self._tflmModel = tflmModel
        self.write()

    def write(self,):
        # path = f"experiments/{deviceName}/{datasetName}/{dt_string}/"
        # if self._dict["config"]["mcu"] == None:
        #     raise Exception("Error")
        mcu = self._dict["config"]["mcu"]
        dataset = self._dict["config"]["dataset"]
        target_lat = self._dict["config"]["target_lat"]
        target_mem = self._dict["config"]["target_mem"]

        path = f"experiments/{mcu}_{dataset}_{target_lat}_{target_mem}_{self._dt_string}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + "record.json", "w") as f:
            json.dump(self._dict, f, default=str)
        if self._keras_model is not None:
            self._keras_model.save(path + "keras.h5")
        if len(self._tflmModel) != 0:
            for (i, tfLitemodel) in enumerate(self._tflmModel):
                with open(path + f"tflmModel_{i}.tflite", "wb") as f:
                    f.write(tfLitemodel.byte_model)

    # def write(self):
    #     now = datetime.now()
    #     dt_string = now.strftime("%d_%m_%Y%H_%M_%S")
    #     # path = f"experiments/{deviceName}/{datasetName}/{dt_string}/"
    #     path = f"experiments/{self.mcu}_{self.dataset}_{self.target_latency}_{self.target_memory}_{dt_string}/"
    #     Path(path).mkdir(parents=True, exist_ok=True)
    #     with open(path + "record.json", "w") as f:
    #         json.dump(self._dict, f)
    #     self._keras_model.save(path + "keras.h5")