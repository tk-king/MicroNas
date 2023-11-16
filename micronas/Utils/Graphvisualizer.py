from random import random, choice
from re import sub
from click import style
from graphviz import Digraph
import string

from matplotlib.pyplot import xlabel
from numpy import choose


class Visualizer():
    def __init__(self, name=None, subgraph=False, label=None):
        if subgraph:
            assert("cluster" in name), "Cluster is not in name"
        self._g = Digraph(name,
            format="pdf",
            edge_attr=dict(fontsize="12", fontname="time"),
            node_attr=dict(style="filled", shape="rect", align="center", fontsize="18",
                           height="0.5", width="0.5", penwidth="2", fontname="times"),
            engine="dot")
        self._g.attr(label=label, labelloc="r")
        self._subgraphs = []
        self._inputs = []
        self._output = None
    
    @property
    def inputOutput(self):
        return self._inputs, self._output

    def addArchWeight(self, label):
        self._g.node(label, color="lightblue")

    def addOp(self, node_name, label):
        self._g.node(node_name, label, color="darkgreen", shape=None)
    
    def addWeight(self, node_name, label):
        self._g.node(node_name, str(label), color="gray")

    def addInput(self, label):
        self._g.node(label, color="yellow")
    
    def addEdge(self, from_node, to_node, label=None):
        self._g.edge(from_node , to_node ,label=label)

    def addWeightEdge(self, form_node, to_node, label=None):
        random_name = ''.join(choice(string.ascii_lowercase) for i in range(20))
        self.addWeight(random_name, label=label)
        self.addEdge(form_node, random_name)
        self.addEdge(random_name, to_node)

    def appendParallel(self, num_layer, layers, weights=None):

        useWeights = weights
        if weights is not None:
            assert(len(weights) == len(layers))
        else:
            useWeights = [0] * len(layers)

        add_name = f"add_{num_layer}"
        if len(layers) > 1:
            self.addOp(add_name, "Add")
        node_names = []
        for n, (l, w) in enumerate(zip(layers, useWeights)):
            node_name = f"{l._get_name()}_{num_layer}_{n}"
            node_label = f"{l._get_name()}\n{l._kernel_size}"
            node_names.append(node_name)
            self.addOp(node_name, label=node_label)
            if len(layers) > 1:
                if weights is not None:
                    self.addWeightEdge(node_name, add_name, w)
                else:
                    self.addEdge(node_name, add_name)
        self._inputs = node_names
        self._output = add_name if len(layers) > 1 else node_name
    
    def appendChReduce(self, num_layer, layers, bottleNeck, choose_weights=None, ch_bt_weights=None, ch_conv_weights=None):
        useWeights = choose_weights is not None
        if choose_weights is None:
            choose_weights = [0] * len(layers)
        else:
            assert(len(choose_weights) == len(layers))

        add_name = f"add_{num_layer}"
        if len(layers) > 1:
            self.addOp(add_name, "Add")

        bt_name = f"{bottleNeck._get_name()}_{num_layer}"
        bt_label = f"{bottleNeck._get_name()}\n{bottleNeck._kernel_size}"
        self.addOp(bt_name, bt_label)
        for n, (l, w) in enumerate(zip(layers, choose_weights)):
            node_name = f"{l._get_name()}_{num_layer}_{n}"
            node_label = f"{l._get_name()}\n{l._kernel_size}"
            self.addOp(node_name, node_label)
            self.addEdge(bt_name, node_name)
            if len(layers) > 1:
                if useWeights:
                    self.addWeightEdge(node_name, add_name, w)
                else:
                    self.addEdge(node_name, add_name)
        self._inputs = [bt_name]
        self._outputs = add_name if useWeights else node_name


        
    def appendLayer(self, subgraph):
        self._subgraphs.append(subgraph)

    def compile(self, input, output):
        for sub_g in self._subgraphs:
            self._g.subgraph(sub_g._g)
        for i in range(len(self._subgraphs) - 1):
            _, layerOut = self._subgraphs[i].inputOutput
            layerIn, _ = self._subgraphs[i + 1].inputOutput
            for i in layerIn:
                self.addEdge(layerOut, i)
        first_input, _ = self._subgraphs[0].inputOutput
        _ , last_output = self._subgraphs[-1].inputOutput
        for f_input in first_input:
            self.addEdge(input, f_input)
        self.addEdge(last_output, output)



# class Visualizer():
#     def __init__(self, g=None):
#         self._g = g
#         if g is None:
#             self._g = Digraph(
#                 format="pdf",
#                 edge_attr=dict(fontsize="20", fontname="time"),
#                 node_attr=dict(style="filled", shape="rect", align="center", fontsize="20",
#                             height="0.5", width="0.5", penwidth="2", fontname="times"),
#                 engine="dot")

#     def addArchWeight(self, label):
#         self._g.node(label, color="lightblue")

#     def addOp(self, label):
#         self._g.node(label, color="darkgreen")

#     def addInput(self, label):
#         self._g.node(label, color="yellow")
    
#     def addEdge(self, from_node, to_node, label=None):
#         self._g.edge(from_node , to_node ,label=label)

#     def appendLayer(self):
#         self._g.subgraph()
