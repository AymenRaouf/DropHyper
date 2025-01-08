import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN
from dhg.random import set_seed

import pandas as pd
import numpy as np
from random import random, seed
import os

def seed_everything(init=0):                                                  
       seed(init)                                                            
       torch.manual_seed(init)                                                      
       torch.cuda.manual_seed_all(init)                                             
       np.random.seed(init)                                                         
       os.environ['PYTHONHASHSEED'] = str(init)                                     
       torch.backends.cudnn.deterministic = True                                    
       torch.backends.cudnn.benchmark = False

def id_mapper(df_col, name, start = 0):
    unique_id = df_col.unique()
    return pd.DataFrame({
            name : unique_id,
            'mappedID' : pd.RangeIndex(start, len(unique_id) + start)
        })

def edge_construction(df1, df2, col, how, right_on, left_on = None):
    if left_on == None:
        left_on = right_on
    links = pd.merge(df1, df2, right_on = right_on, left_on = left_on, how = how)
    links = torch.from_numpy(links[col].values)
    return links

def hyperedge_construction(edges):
    hyperedges = []
    current_hyperedge = []
    for i in range(len(edges[0]) - 1):
        if i not in current_hyperedge:
              current_hyperedge.append(int(edges[0, i]))
        current_hyperedge.append(int(edges[1, i]))
        if edges[0, i] != edges[0, i+1]:
                hyperedges.append(tuple(set(current_hyperedge)))
                current_hyperedge = []
    if len(current_hyperedge) != 0:
        hyperedges.append(tuple(set(current_hyperedge)))  
             
    return hyperedges



def load_data(dataset):
    seed_everything()
    methods = 'embedd-er' #embedd-er or BERT
    class_type = 'dct' #dct or rdfs
    class_file = class_type + ".csv"

    script_dir = os.path.dirname(os.path.realpath('__file__'))
    data_path = os.path.join(script_dir, './Data/' + dataset + '/data/')
    embeddings_path = os.path.join(script_dir, './Data/' + dataset + '/embeddings/')
    precedence_path = os.path.join(script_dir, './Data/' + dataset + '/precedence/')

    df_chapters = pd.read_csv(data_path + 'chapters.csv', delimiter = '|')
    df_chapters_embeddings = pd.read_csv(embeddings_path + 'chapters_' + methods +'.csv', delimiter = '|', index_col=0)
    df_concepts = pd.read_csv(data_path + 'concepts.csv', delimiter = '|')
    df_concepts_embeddings = pd.read_csv(embeddings_path + 'concepts_' + methods +'.csv', delimiter = '|', index_col=0)
    df_classes = pd.read_csv(data_path + 'classes/rdfs.csv', delimiter = '|')
    df_classes_embeddings = pd.read_csv(embeddings_path + '/classes/' + class_type + '_' + methods +'.csv', delimiter = '|')
    df_precedences_episodes = pd.read_csv(precedence_path + 'precedences_episodes.csv', delimiter = '|')
    df_precedences_series = pd.read_csv(precedence_path + 'precedences_series.csv', delimiter = '|')

    df_concepts['Concept'] = df_concepts['Concept'].apply(lambda x : x.split('/')[-1])

    df_classes = df_classes.dropna()
    print(f'{df_chapters["Cid"].isna().sum().sum():04d} NaN values in chapters.')
    print(f'{df_concepts.isna().sum().sum():04d} Nan values in concepts.')
    print(f'{df_classes.isna().sum().sum():04d} Nan values in classes.')
    print(f'{df_precedences_episodes.isna().sum().sum():04d} Nan values in episdes precedences.')
    print(f'{df_precedences_series.isna().sum().sum():04d} Nan values in series precedences.')

    print(df_chapters.shape)



    unique_oer_id = id_mapper(df_chapters['Cid'], 'OER')
    oer_ids_size = len(unique_oer_id['mappedID'].values)

    unique_concept_id =  id_mapper(df_concepts['Concept'], 'Concept', oer_ids_size)
    concept_ids_size = len(unique_concept_id['mappedID'].values)
    
    unique_class_id =  id_mapper(df_classes['Class'], 'Class', oer_ids_size + concept_ids_size)
    class_ids_size = len(unique_class_id['mappedID'].values)

    oer_covers_concept_subject = edge_construction(df1 = df_concepts, df2 = unique_oer_id, col = 'mappedID', 
                                       how = 'left', right_on = 'OER')
    oer_covers_concept_pr = edge_construction(df1 = df_concepts, df2 = unique_oer_id, col = 'PR', 
                                            how = 'right', right_on = 'OER')
    oer_covers_concept_object = edge_construction(df1 = df_concepts, df2 = unique_concept_id, col = 'mappedID', 
                                        how = 'left', right_on = 'Concept')

    oer_before_oer_ep_subject = edge_construction(df1 = df_precedences_episodes, df2 = unique_oer_id, col = 'mappedID', 
                                    how = 'left', left_on = 'Before', right_on = 'OER')
    oer_before_oer_ep_object = edge_construction(df1 = df_precedences_episodes, df2 = unique_oer_id, col = 'mappedID', 
                                    how = 'left', left_on = 'After', right_on = 'OER')
    oer_before_oer_sr_subject = edge_construction(df1 = df_precedences_series, df2 = unique_oer_id, col = 'mappedID', 
                                    how = 'left', left_on = 'Before', right_on = 'OER')
    oer_before_oer_sr_object = edge_construction(df1 = df_precedences_series, df2 = unique_oer_id, col = 'mappedID', 
                                    how = 'left', left_on = 'After', right_on = 'OER')

    concept_belongs_class_subject = edge_construction(df1 = df_classes, df2 = unique_concept_id, col = 'mappedID', 
                                    how = 'left', left_on = 'Concept', right_on = 'Concept')
    concept_belongs_class_object = edge_construction(df1 = df_classes, df2 = unique_class_id, col = 'mappedID', 
                                    how = 'left', left_on = 'Class', right_on = 'Class')

    oer_covers_concept = torch.stack([oer_covers_concept_subject, oer_covers_concept_object], dim = 0)
    oer_covers_concept_rev = torch.stack([oer_covers_concept_object, oer_covers_concept_subject], dim = 0)
    oer_before_oer_ep = torch.stack([oer_before_oer_ep_subject, oer_before_oer_ep_object], dim = 0)
    oer_before_oer_sr = torch.stack([oer_before_oer_sr_subject, oer_before_oer_sr_object], dim = 0)
    concept_belongs_class = torch.stack([concept_belongs_class_subject, concept_belongs_class_object], dim = 0)
    concept_belongs_class_rev = torch.stack([concept_belongs_class_object, concept_belongs_class_subject], dim = 0)
    print(oer_covers_concept.shape)
    print(oer_covers_concept_rev.shape)
    print(oer_before_oer_ep.shape)
    print(oer_before_oer_sr.shape)
    print(concept_belongs_class.shape)
    print(concept_belongs_class_rev.shape)


    chapters_embeddings_tmp = {}
    concepts_embeddings_tmp = {}
    classes_embeddings_tmp = {}

    chapters_r = range(len(df_chapters['Cid'].unique()))
    concepts_c = range(len(df_concepts['Concept'].unique()))
    classes_c = range(len(df_classes['Class'].unique()))

    if methods == 'BERT':
        entity_features = 768
    elif methods == 'embedd-er':
        entity_features = 300

    chapters_embeddings = np.zeros(shape=(len(chapters_r), entity_features))
    concepts_embeddings = np.zeros(shape=(len(concepts_c), entity_features))
    classes_embeddings = np.zeros(shape=(len(classes_c), entity_features))


    i = 0
    for r in chapters_r:
        chapters_embeddings_tmp[r] = list(filter(None, df_chapters_embeddings['Chapters Embeddings'][r].strip("[]\n").replace("'","").split(" ")))
        chapters_embeddings_tmp[r] = [float(f) for f in chapters_embeddings_tmp[r]]
        for a in range(len(chapters_embeddings_tmp[r])):
                chapters_embeddings[i][a] = chapters_embeddings_tmp[r][a]
        i += 1

    i = 0
    for r in concepts_c:
        concepts_embeddings_tmp[r] = list(filter(None, df_concepts_embeddings['Concepts Embeddings'][r].strip("[]\n").replace("'","").split(" ")))
        concepts_embeddings_tmp[r] = [float(f) for f in concepts_embeddings_tmp[r]]
        for a in range(len(concepts_embeddings_tmp[r])):
                concepts_embeddings[i][a] = concepts_embeddings_tmp[r][a]
        i += 1   

    i = 0
    for r in classes_c:
        classes_embeddings_tmp[r] = list(filter(None, df_classes_embeddings['Classes Embeddings'][r].strip("[]\n").replace("'","").split(" ")))
        classes_embeddings_tmp[r] = [float(f) for f in classes_embeddings_tmp[r]]
        for a in range(len(classes_embeddings_tmp[r])):
                classes_embeddings[i][a] = classes_embeddings_tmp[r][a]
        i += 1

    chapters_embeddings = torch.from_numpy(chapters_embeddings).to(torch.float32)
    concepts_embeddings = torch.from_numpy(concepts_embeddings).to(torch.float32)
    classes_embeddings = torch.from_numpy(classes_embeddings).to(torch.float32)



    return {
         'ids' : {
              'oer' : unique_oer_id['mappedID'].values,
              'concept' : unique_concept_id['mappedID'].values,
              'class' : unique_class_id['mappedID'].values
         },
         'features' : {
              'oer' : chapters_embeddings,
              'concept' : concepts_embeddings,
              'class' : classes_embeddings
         },
         'relations' : {
              'oer_concept' : oer_covers_concept,
              #'concept_oer' : oer_covers_concept_rev,
              'oer_sr_oer' : oer_before_oer_sr,
              'oer_ep_oer' : oer_before_oer_ep,
              'concept_class' : concept_belongs_class,
              #'class_concept' : concept_belongs_class_rev
         }
    }


def concatenate_data(dataset):
    ids = torch.cat([
        torch.from_numpy(dataset['ids']['oer']).to(torch.float32), 
        torch.from_numpy(dataset['ids']['concept']).to(torch.float32), 
        torch.from_numpy(dataset['ids']['class']).to(torch.float32)], 
        dim = 0).to(torch.long)
    features = torch.cat([dataset['features']['oer'], dataset['features']['concept'], dataset['features']['class']], dim = 0)
    relations = torch.cat([
        dataset['relations']['oer_concept'], 
        #dataset['relations']['concept_oer'],
        dataset['relations']['oer_sr_oer'],
        dataset['relations']['oer_ep_oer'],
        #dataset['relations']['class_concept'],
        dataset['relations']['concept_class']],
        dim = 1)
    
    return {
         'ids' : ids,
         'features' : features,
         'relations' : relations
    }


def cross_validation_splits(publisher):
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    data_path = os.path.join(script_dir, './Data/' + publisher + '/data/')
    df_cross_validation = pd.read_csv(data_path + 'cross_validation.csv', delimiter = ',')
    cv = df_cross_validation['chunk'].nunique()
    train_mask_in = []
    train_mask_out =  []
    train_target = []
    test_mask_in = []
    test_mask_out = []
    test_target = []
    for i in range(cv):
        train_mask_in.append(df_cross_validation[(df_cross_validation['chunk'] == i) & (df_cross_validation['train'] == 1)]['in'].values)
        train_mask_out.append(df_cross_validation[(df_cross_validation['chunk'] == i) & (df_cross_validation['train'] == 1)]['out'].values)
        train_target.append(df_cross_validation[(df_cross_validation['chunk'] == i) & (df_cross_validation['train'] == 1)]['label'].values)
        test_mask_in.append(df_cross_validation[(df_cross_validation['chunk'] == i) & (df_cross_validation['train'] == 0)]['in'].values)
        test_mask_out.append(df_cross_validation[(df_cross_validation['chunk'] == i) & (df_cross_validation['train'] == 0)]['out'].values)
        test_target.append(df_cross_validation[(df_cross_validation['chunk'] == i) & (df_cross_validation['train'] == 0)]['label'].values)

    return {
        'cv' : cv,
        'train_mask_in' : train_mask_in,
        'train_mask_out' : train_mask_out,
        'train_target' : train_target,
        'test_mask_in' : test_mask_in,
        'test_mask_out' : test_mask_out,
        'test_target' : test_target
    }



def hg_from_data(publisher:str) -> Hypergraph:
     
    data = load_data(publisher)
    data = concatenate_data(data)
    X, labels, relations = data['features'], data["ids"], data['relations']
    relations = hyperedge_construction(relations)
    data_masks = cross_validation_splits('OYC')
    return X, labels, relations, data_masks


def hg_for_cv(i:int, masks:list, labels:list, relations:list)->Hypergraph:
    test_relations = []
    test_size = len(masks['test_mask_in'][i])
    for j in range(test_size):
        if masks['test_target'][i][j] == 1:
            test_relations.append((masks['test_mask_in'][i][j], masks['test_mask_out'][i][j]))
    relations = [r for r in relations if r not in test_relations]
    return Hypergraph(len(labels), relations)

def dropout_hgnn(method:str, rate:float, labels:list, relations:list)->Hypergraph:

    #Dropping nodes
    if method == 'dropnode' and rate != 0.0:
        non_droped_labels = [l for l in labels if random() > rate]
        non_droped_labels = torch.Tensor(non_droped_labels).long()
        non_droped_relations = []
        for relation in relations:
            non_droped_relations.append(tuple(x for x in relation if x not in non_droped_labels))
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

    return Hypergraph(len(labels), relations)


def train_hgnn(net, X, G, train_idx, epoch):
    net.train()

    st = time.time()
    #optimizer.zero_grad()
    outs = net(X, G)
    #outs, lbls = outs[train_idx], lbls[train_idx]
    #loss = F.cross_entropy(outs, lbls)
    #loss.backward()
    #optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}")
    return outs





def train_mlp(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
