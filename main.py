from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import train, infer, dropout_hgnn, load_dataset, dirichlet_energy
from models import *
import pandas as pd

from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

if __name__ == "__main__":
    drop_rates = [0.05, 0.1, 0.2, 0.5, 0.8]
    drop_methods = ['dropnode', 'dropedge', 'drophyperedge']
    depths = [1, 2, 4, 6, 8]
    epochs = 500

    selected_params = [{
        'depth' : depth,
        'drop_method': drop_method,
        'drop_rate' : drop_rate
    } for drop_rate in drop_rates
    for drop_method in drop_methods
    for depth in depths]

    

    set_seed(0)



    model_names = ['HGNN', 'HGNNP', 'UniSAGE', 'UniGCN', 'UniGAT']
    dataset_names = ['CocitationCora', 'CocitationCiteseer', 'CoauthorshipCora', 'CoauthorshipDBLP', 'CocitationPubmed', 'Cooking200', 'Tencent2k']

    df_results = pd.DataFrame()


    for dataset in dataset_names:
        data = load_dataset(dataset)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator(["accuracy"])
        # Add code to check for features before using the adjacency matrix
        try:
            X, lbl = data["features"], data["labels"]  
        except AssertionError as e:
            X, lbl = torch.eye(data["num_vertices"]), data["labels"]

        ids = list(range(data['num_vertices']))
        relations = data["edge_list"]
        
        #G = Hypergraph(data["num_vertices"], data["edge_list"])
        
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        for model in model_names:
            for params in selected_params:
                depth = params['depth']
                drop_method = params['drop_method']
                init_drop_rate = params['drop_rate']
                epochs = params['epochs']

                net = load_model(model, X, data, depth)
                optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
                X, lbl = X.to(device), lbl.to(device)
                G = Hypergraph(data["num_vertices"], data["edge_list"])
                G = G.to(device)
                net = net.to(device)

                best_state = None
                best_epoch, best_val = 0, 0

                for epoch in range(epochs): 
                    
                    print(len(relations), G, G.L_HGNN.shape, X.shape)

                    if epoch == 0:
                        energy = dirichlet_energy(G,X)
                        print(f"Dirichet energy : {energy}")
                    drop_rate = (energy) * init_drop_rate

                    print(f"Old drop rate : {init_drop_rate} New drop rate : {drop_rate}")
                    # train
                    G = dropout_hgnn(drop_method, drop_rate, ids, relations)
                    loss_item, energy = train(net, X, G, lbl, train_mask, optimizer, epoch)

                    # validation
                    if epoch % 10 == 0:
                        with torch.no_grad():
                            val_res = infer(net, X, G, lbl, val_mask, evaluator)
                        if val_res > best_val:
                            print(f"update best: {val_res:.5f}")
                            best_epoch = epoch
                            best_val = val_res
                            best_state = deepcopy(net.state_dict())
                    print("\n")
                print("\ntrain finished")
                print(f"best val: {best_val:.5f}")
                # test
                print("test...")
                net.load_state_dict(best_state)        
                res = infer(net, X, G, lbl, test_mask, evaluator, test=True)
                print(f"final result: epoch: {best_epoch}")
                print(res)
                results = {
                    'model' : model,
                    'data' : dataset,
                    'depth' : depth,
                    'epochs' : epochs,
                    'val_accuracy' : best_val,
                    'test_accuracy' : res['accuracy'],
                    'drop_method': drop_method,
                    'drop_rate' : drop_rate,
                    
                }
                df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)

    df_results.to_csv('results_are_here.csv')