from dhg.data import Cooking200, CocitationCora, CoauthorshipCora, CocitationPubmed, CoauthorshipDBLP, CocitationCiteseer, News20, Yelp3k, Tencent2k
from random import random, seed
import torch.nn.functional as F
from dhg import Hypergraph
import time
import torch


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
    M_min = dir_matrix.min()
    M_max = dir_matrix.max()
    dir_matrix_normalized = (dir_matrix - M_min) / (M_max - M_min)
    return dir_matrix_normalized.trace()/dir_matrix_normalized.shape[0]


def dropout_hgnn(method:str, rate:float, nodes:list, relations:list)->Hypergraph:    
    # Dropping nodes
    if method == 'dropnode' and rate != 0.0:
        non_droped_nodes = [l for l in nodes if random() > rate]
        non_droped_nodes = torch.Tensor(non_droped_nodes).long()
        non_droped_relations = []
        for relation in relations:
            non_droped_relations.append(tuple(x for x in relation if x not in non_droped_nodes))
        relations = non_droped_relations

    # Dropping edges
    if method == 'dropedge' and rate != 0.0:
        non_droped_relations = []
        for relation in relations:
            non_droped_relations.append(tuple(x for x in relation if random() > rate))
        relations = non_droped_relations

    # Dropping hyperedges
    if method == 'drophyperedge' and rate != 0.0:
        relations = [r for r in relations if random() > rate]

    return Hypergraph(len(nodes), relations)



def forward(net, X, G, epoch):
    
    st = time.time()
    outs = net(X, G)
    energy = torch.sigmoid(dirichlet_energy(G, outs))
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, message passing...")
    return energy, outs



def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net.predict(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, evaluator, test=False):
    net.eval()
    outs = net.predict(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res