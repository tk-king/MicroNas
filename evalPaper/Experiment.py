from micronas import MicroNas
from main import micro_nas_search
import os
import argparse

BASE_DIR = "evalPaper/experiments"

class Experiment():
    def __init__(self, model, dataset, cv, search_strategy, mcu, target_lat, target_mem):
        assert model in ["micronas"]
        self.model = model
        self.dataset = dataset
        self.search_strategy = search_strategy
        self.mcu = mcu
        self.target_lat = target_lat
        self.target_mem = target_mem
        self.cv = cv
        self.models = []

    
    def run(self):
        if self.model == "micronas":
            self.models = micro_nas_search(self.dataset, self.cv, self.search_strategy, self.mcu, self.target_lat, self.target_mem)



    def save(self):
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        for model in self.models:
            model.save(BASE_DIR)


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


    exp = Experiment(args.model, args.dataset, args.cv, args.search_strategy, args.mcu, args.target_lat, args.target_mem)
    exp.run()
    exp.save()

