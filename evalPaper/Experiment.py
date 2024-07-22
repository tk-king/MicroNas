from micronas import MicroNas
from main import micro_nas_search
import os
import argparse
import json
import datetime

BASE_DIR = "evalPaper/experiments"

class Experiment():
    def __init__(self, args):
        assert args.model in ["micronas"]
        self.model = args.model
        self.dataset = args.dataset
        self.search_strategy = args.search_strategy
        self.mcu = args.mcu
        self.target_lat = args.target_lat
        self.target_mem = args.target_mem
        self.cv = args.cv
        self.models = []
        self.args = args
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.dir = os.path.join(BASE_DIR, f"{timestamp}")

    
    def run(self):
        if self.model == "micronas":
            self.models = micro_nas_search(self.dataset, self.cv, self.search_strategy, self.mcu, self.target_lat, self.target_mem)



    def save(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        for model in self.models:
            model.save(self.dir)
        args_dict = vars(self.args)
        with open(os.path.join(self.dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--dataset", type=str)
    argparser.add_argument("--cv", type=int)
    argparser.add_argument("--search_strategy", type=str)
    argparser.add_argument("--mcu", type=str)
    argparser.add_argument("--target_lat", type=float, default=None)
    argparser.add_argument("--target_mem", type=float, default=None)
    args = argparser.parse_args()


    exp = Experiment(args)
    exp.run()
    exp.save()

