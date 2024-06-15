
from sponet import CNVM
from sponet import CNVMParameters
import numpy as np
import networkx as nx

num_nodes = 1000

r = np.array([[0, .8], [.2, 0]])
r_tilde = np.array([[0, .1], [.2, 0]])

#network = nx.erdos_renyi_graph(n=num_nodes, p=0.1)

network = nx.complete_graph(num_nodes)

params = CNVMParameters(
    num_opinions=2,
    network=network,
    r=r,
    r_tilde=r_tilde,
)


x_init = np.random.randint(0, 2, num_nodes)
model = CNVM(params)
t, x = model.simulate(t_max=50, x_init=x_init)

print(t,x)