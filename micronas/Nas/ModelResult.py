from dataclasses import dataclass, field
from keras.models import Model
from typing import Dict
import json
import os
from micronas.Utils.JsonEncoder import JsonEncoder

@dataclass
class ModelResult:
    keras_model: Model
    tf_lite_model: object

    latency: float
    memory: float

    config: Dict

    eval_keras: Dict = field(default_factory=dict)
    eval_tf_lite: Dict = field(default_factory=dict)
    

    def save(self, path: str):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the Keras model
        model_path = os.path.join(path, 'model.h5')
        self.keras_model.save(model_path)

        # Save the TFLite model
        tflite_path = os.path.join(path, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(self.tf_lite_model)
        
        # Save the other attributes
        result_data = {
            'latency': self.latency,
            'memory': self.memory,
            'eval_keras': self.eval_keras,
            'eval_tf_lite': self.eval_tf_lite,
            'config': self.config
        }

        for key, value in result_data.items():
            print(type(value))
        
        # Save the attributes as a JSON file
        json_path = os.path.join(path, 'result.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4, cls=JsonEncoder)
