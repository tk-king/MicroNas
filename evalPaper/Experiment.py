from main import micro_nas_search
import os
import argparse
import json
import datetime
from othermodels.trainEval import train_eval
import logging
from typing import List
from micronas.Nas.ModelResult import ModelResult
from micronas.config import MicroNasMCU, Config
from micronas.Utils.JsonEncoder import JsonEncoder

BASE_DIR = "evalPaper/experiments"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Experiment():
    def __init__(self, args):
        self.model = args.model
        self.dataset = args.dataset
        self.search_strategy = args.search_strategy
        self.mcu = args.mcu
        self.target_lat = args.target_lat
        self.target_mem = args.target_mem
        self.cv = args.cv
        self.models : List[ModelResult] = []
        self.args = args
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        self.dir = os.path.join(BASE_DIR, f"{timestamp}")

    
    def run(self):
        if self.model == "micronas":
            self.models = micro_nas_search(self.dataset, self.cv, self.search_strategy, self.mcu, self.target_lat, self.target_mem)
        else:
            self.models = [train_eval(self.dataset, self.cv, self.model)]
        


    def save(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        for i, model in enumerate(self.models):
            model.save(self.dir, i)
        args_dict = vars(self.args)
        with open(os.path.join(self.dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=4, cls=JsonEncoder)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--dataset", type=str)
    argparser.add_argument("--cv", type=int)
    argparser.add_argument("--search_strategy", type=str)
    argparser.add_argument("--mcu", type=MicroNasMCU)
    argparser.add_argument("--target-lat", type=float, default=None)
    argparser.add_argument("--target-mem", type=float, default=None)
    args = argparser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment with args: {args}")
    exp = Experiment(args)
    exp.run()
    exp.save()

