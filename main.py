from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import train, infer, dropout_hgnn, load_dataset, dirichlet_energy
from models import *
import pandas as pd
import sys
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

if __name__ == "__main__":

    set_seed(0)
    drop_rates = [0.05, 0.1, 0.2, 0.5, 0.8]
    drop_rate = 0.0
    drop_methods = ['no dropout', 'dropnode', 'dropedge', 'drophyperedge']
    depths = [1, 2, 4, 8, 12]
    epochs = 500

    dataset = sys.argv[1]
    model = sys.argv[2]

    selected_params = [{
        'depth' : depth,
        'drop_method': drop_method,
        'drop_rate' : drop_rate
    } for drop_rate in drop_rates
    for drop_method in drop_methods
    for depth in depths]

    print(f"Script running model {model} on dataset {dataset}...")
    

    #model_names = ['HGNNP', 'UniSAGE', 'UniGCN', 'UniGIN', 'UniGAT', 'HNHN']
    #dataset_names = ['CocitationCora', 'CocitationCiteseer', 'CoauthorshipCora', 'CoauthorshipDBLP', 'CocitationPubmed', 'Cooking200', 'Tencent2k']

    df_results = pd.DataFrame()

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
    
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

     
    for params in selected_params:
        depth = params['depth']
        drop_method = params['drop_method']
        init_drop_rate = params['drop_rate']
        oversmoothing = []
        loss_values = []
        val_accuracy = []
        
        if drop_method == "no dropout" and init_drop_rate > 0.05 :
            continue


        net = load_model(model, X, data, depth)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        X, lbl = X.to(device), lbl.to(device)
        G = Hypergraph(data["num_vertices"], data["edge_list"])
        G = G.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0

        for epoch in range(epochs): 
            
            set_seed(epoch)                   
            #drop_rate = torch.tanh(energy) * init_drop_rate
            
            # train
            G = dropout_hgnn(drop_method, drop_rate, ids, relations)
            loss_item = train(net, X, G, lbl, train_mask, optimizer, epoch)

            # validation
            if epoch % 10 == 0:
                with torch.no_grad():
                    val_res, energy = infer(net, X, G, lbl, val_mask, evaluator)
                    drop_rate = torch.exp(-3 * energy) * init_drop_rate
                    print(f"Oversmoothing measure : {energy}")
                    print(f"Old drop rate : {init_drop_rate} New drop rate : {drop_rate}")
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
                val_accuracy.append(val_res)
                oversmoothing.append(energy.item())
                loss_values.append(loss_item)
            if epoch % 20 == 0:
                
                res, _ = infer(net, X, G, lbl, test_mask, evaluator, test=True)
                results = {
                    'model' : model,
                    'data' : dataset,
                    'depth' : depth,
                    'epochs' : epoch,
                    'best_val_accuracy' : best_val,
                    'test_accuracy' : res['accuracy'],
                    'drop_method': drop_method,
                    'drop_rate' : init_drop_rate,
                    'oversmoothing' : oversmoothing,
                    'val_accuracy' : val_accuracy,
                    'loss' : loss_values
                }
                df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)
            

        print("\ntrain finished")
        print(f"best val: {best_val:.5f}")
        # test
        print("test...")
        net.load_state_dict(best_state)        
        res, _ = infer(net, X, G, lbl, test_mask, evaluator, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(f"dataset {dataset}")
        print(f"model {model}")
        print(f"parameters used: {params}")
        print(res)
        results = {
            'model' : model,
            'data' : dataset,
            'depth' : depth,
            'epochs' : epoch+1,
            'best_val_accuracy' : best_val,
            'test_accuracy' : res['accuracy'],
            'drop_method': drop_method,
            'drop_rate' : init_drop_rate,
            'oversmoothing' : oversmoothing,
            'val_accuracy' : val_accuracy,
            'loss' : loss_values
        }

        df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)
    df_results.to_csv(f'results_{dataset}_{model}.csv')