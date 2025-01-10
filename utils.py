from dhg.data import Cooking200, CocitationCora, CoauthorshipCora, CocitationPubmed, CoauthorshipDBLP, CocitationCiteseer, News20, Yelp3k, Tencent2k
from random import random, seed
import torch.nn.functional as F
from dhg import Hypergraph
import time
import torch
import matplotlib.pyplot as plt


def load_dataset(name):
    match name:
        case 'CocitationCora':
            return CocitationCora()
        case 'CoauthorshipCora':
            return CoauthorshipCora()
        case 'CoauthorshipDBLP':
            return CoauthorshipDBLP()
        case 'CocitationCiteseer':
            return CocitationCiteseer()
        case 'CocitationPubmed':
            return CocitationPubmed()
        case 'Cooking200':
            return Cooking200()
        case 'News20':
            return News20()
        case 'Yelp3k':
            return Yelp3k()
        case 'Tencent2k':
            return Tencent2k()
    


def dirichlet_energy(G,X):
    dir_matrix = torch.mm(X.T, torch.mm(G.L_HGNN, X))
    return dir_matrix.trace()/len(dir_matrix)

def dirichlet_energy_explicit(G, X):
    total_v = 0
    total_diff = 0
    num_v = G.num_v
    #print(G.deg_v)
    for v in range(num_v):
        e_neigh = G.nbr_e(v)
        if len(e_neigh) != 0:
            for e in e_neigh:
                v_neigh = G.nbr_v(e)
                for v_n in v_neigh:
                    total_v += 1
                    diff = X[v] - X[v_n]
                    distance = torch.sqrt(torch.sum(diff**2))
                    total_diff += distance
    return total_diff/total_v


def dropout_hgnn(method:str, rate:float, nodes:list, relations:list)->Hypergraph:    
    # Dropping nodes
    if method == 'dropnode' and rate != 0.0:
        non_droped_nodes = [l for l in nodes if random() > rate]
        non_droped_nodes = torch.Tensor(non_droped_nodes).long()
        non_droped_relations = []
        for relation in relations:
            non_droped_relations.append(tuple(x for x in relation if x not in non_droped_nodes))
        new_relations = non_droped_relations

    # Dropping edges
    if method == 'dropedge' and rate != 0.0:
        non_droped_relations = []
        for relation in relations:
            non_droped_relations.append(tuple(x for x in relation if random() > rate))
        new_relations = non_droped_relations

    # Dropping hyperedges
    if method == 'drophyperedge' and rate != 0.0:
        new_relations = [r for r in relations if random() > rate]

    if method == "no dropout":
        new_relations = relations

    return Hypergraph(len(nodes), new_relations)


def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)

    # Show the plot
    plt.show()
    #energy = torch.sqrt(dirichlet_energy(G, outs))
    
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, evaluator, test=False):
    net.eval()
    outs = net(X, G)
    energy_explicit = torch.sqrt(dirichlet_energy_explicit(G, outs))
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res, energy_explicit


def column2array(column):
    values = list(filter(None, column.strip("[]\n").replace(",","").split(" ")))
    values = [float(f) for f in values]
    return values