import torch
from torch import nn
from torch.autograd import Variable
from micronas.Nas.Utils import AvgrageMeter, accuracy
from tqdm import tqdm

from micronas.config import Config


class SearchLogger():
    def __init__(self):
        self._loss_out = []
        self._loss_lat = []
        self._loss_mem = []
        self._latency = []
        self._memory = []

    def appendLoss(self, loss_out, loss_lat, loss_mem):
        self._loss_out.append(float(loss_out.detach().numpy()))
        self._loss_lat.append(float(loss_lat.detach().numpy()))
        self._loss_mem.append(float(loss_mem.detach().numpy()))

    def appendLatency(self, latency):
        self._latency.append(float(latency.detach().numpy()))

    def appendMemory(self, memory):
        self._memory.append(float(memory.detach().numpy()))


class ArchLoss(nn.Module):
    def __init__(self, target_lat, target_mem, lr_lat, lr_mem, logger, nas_weights, max_epoch) -> None:
        super(ArchLoss, self).__init__()
        self.cls_loss = nn.NLLLoss()
        self._target_lat = target_lat
        self._target_mem = target_mem

        self._lr_lat = torch.tensor(lr_lat)
        self._lr_mem = torch.tensor(lr_mem)
        self._logger = logger
        self._nas_weights = nas_weights

        self._max_epoch = max_epoch


    def forward(self, out, target, epoch):
        pred, latency, memory = out
        mean_lat = torch.mean(latency)
        mean_mem = torch.mean(memory)


        self._logger.appendLatency(mean_lat)
        self._logger.appendMemory(mean_mem)

        loss_lat = torch.tensor(0)
        loss_mem = torch.tensor(0)


        if self._target_lat:
            mul_lat = 4 if self._target_lat < mean_lat else 0.002
            loss_lat = mul_lat * torch.log(mean_lat / self._target_lat)

        if self._target_mem:
            mul_mem = 4 if self._target_mem < mean_mem else 0.002
            loss_mem = mul_mem * torch.log(mean_mem / self._target_mem)

        cls_loss = self.cls_loss(pred, target)
        self._logger.appendLoss(cls_loss, loss_lat, loss_mem)


        loss = cls_loss + loss_lat + loss_mem

        return loss, cls_loss, loss_lat, loss_mem, mean_lat, mean_mem


class DNasStrategy():

    def __init__(self, network) -> None:
        self.network = network
        self.arch_optim = torch.optim.SGD(network.get_nas_weights(), lr=0.36)
        self.arch_schedular = None
        self.criterion = None
        self.optimizer = torch.optim.Adam(network.parameters())
        self._alpha_lat = 1
        self._alpha_mem = 1
        self._arch_loss = None
        self._logger = SearchLogger()

    def visualize(self, e_len):
        self._logger.visualize(e_len)

    def search(self, train_queue, valid_queue, target_lat, target_mem, callback=None, num_epochs=20, epochs_pretrain=0, num_arch_train_steps=1, eps_decay=0.995, alpha_lat=1, alpha_mem=1):
        
        if callback is None:
            callback = lambda x: None
        
        self.criterion = nn.NLLLoss()
        self.arch_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(self.arch_optim, T_max=len(train_queue) * Config.search_epochs, eta_min=0.0008)


        assert num_epochs > epochs_pretrain
        # self.network.set_alpha_grad(True)
        objs = AvgrageMeter()
        top1 = AvgrageMeter()

        self._nas_weights = self.network.get_nas_weights()

        self._eps_decay = eps_decay
        self._arch_loss = ArchLoss(target_lat, target_mem, alpha_lat, alpha_mem, self._logger, self._nas_weights, num_epochs)

        self.network.train()
        num_steps = len(train_queue)
        ctr = 0
        for epoch in range(num_epochs):
            print("Epoch: ", epoch + 1, " / ", num_epochs)
            objs.reset()
            top1.reset()
            for step, (input_time, input_freq, target) in enumerate(pbar := tqdm(train_queue)):
                # input_time = torch.swapaxes(input_time, 1, 2)
                self.network.train()

                n = input_time.size(0)

                input = Variable(input_time, requires_grad=False).float().to(Config.compute_unit)
                target = Variable(target, requires_grad=False).to(Config.compute_unit)

                # Update the architecture
                for _ in range(num_arch_train_steps):
                    if epoch - epochs_pretrain >= 0:
                        input_search_time, input_search_freq, target_search = next(
                            iter(valid_queue))
                        input_search = Variable(
                            input_search_time, requires_grad=False).float()
                        target_search = Variable(
                            target_search, requires_grad=False).to(Config.compute_unit)
                        self.arch_optim.zero_grad()
                        # input_search = torch.unsqueeze(input_search, dim=1)
                        # print(input_search.shape)
                        output = self.network(input_search)
                        loss, loss_ce, loss_lat, loss_mem, mean_lat, mean_mem = self._arch_loss(output, target_search, epoch)
                        # loss = loss_ce + loss_lat + loss_mem
                        postfix = {
                        "Loss": round(float(loss.cpu().detach().numpy()), 4), 
                        "Loss_CE": round(float(loss_ce.cpu().detach().numpy()), 4), 
                        "Loss_LAT": round(float(loss_lat.cpu().detach().numpy()), 4), 
                        "Loss_MEM": round(float(loss_mem.cpu().detach().numpy()), 4), 
                        "eps": self.network._t, 
                        "mean_lat": round(float(mean_lat.cpu().detach().numpy()), 4), 
                        "mean_mem": mean_mem.cpu().detach().numpy()}
                        pbar.set_postfix(postfix)
                        loss.backward()
                        # nn.utils.clip_grad_norm_(self._nas_weights, 0.1)
                        # for w_g in self._nas_weights:
                        #     print(w_g.grad)
                        #     print(w_g)
                        #     print("--------")
                        self.arch_optim.step()
                        self.arch_schedular.step()

                # Update the network parameters
                self.optimizer.zero_grad()
                # input = torch.unsqueeze(input, dim=1)
                logits, _, _ = self.network(input)
                loss = self.criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5)
                self.optimizer.step()

                [prec1] = accuracy(logits, target, topk=[1])
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                self.network._t = max(self.network._t * self._eps_decay, 1e-5)
                ctr += 1
                callback((ctr * 100) / (num_epochs * num_steps))

            # self.network._t = max(self.network._t * esp_update_step, 1e-5)
            print('Step: %03d, Obj_avg: %e, Top1_avg: %f' %
                  (step, objs.avg, top1.avg))


# Evaluate the supernet
def infer(test_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top3 = AvgrageMeter()
    model.eval()

    for step, (input_search_time, input_search_freq, target_search) in enumerate(test_queue):
        input = torch.swapaxes(input_search_time, 1, 2)
        input = Variable(input, volatile=True).float()
        target = Variable(target_search, volatile=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top3.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #   print('test %03d %e %f %f' % (step, objs.avg, top1.avg, top3.avg))

    return top1.avg, objs.avg
