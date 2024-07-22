import torch
import json
from micronas import MicroNas, MicroNasMCU
from micronas.config import DefaultConfig


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, MicroNasMCU):
            return obj.name
        elif isinstance(obj, DefaultConfig):
            return obj.to_serializable_dict()
        return super(JsonEncoder, self).default(obj)
