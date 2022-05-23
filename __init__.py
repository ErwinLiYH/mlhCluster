from . import k_cluster
from Kkit import ksort
import pandas as pd
from sklearn import metrics

def two_layers_cluster(linkage, outer_distence_threshold, inner_distence_threshold):
    if outer_distence_threshold >= inner_distence_threshold:
        raise Exception("outer_distence_threshold must small than inner_distence_threshold")
    root, Node_list = k_cluster.to_tree(linkage)
    k_cluster.compress_tree(Node_list, outer_distence_threshold)
    outer_relation = k_cluster.gen_classes(Node_list,outer_distence_threshold)
    k_cluster.clean_tree(Node_list,outer_distence_threshold)
    k_cluster.compress_tree(Node_list, inner_distence_threshold)
    inner_relation = k_cluster.gen_classes(Node_list, inner_distence_threshold)  
    return outer_relation, inner_relation

def __search_outer_and_inner_relation(outer_relation, inner_relation, key):
    for k,v in outer_relation.items():
        if key in v and k != v:
            return ["outer", k, key]
    for k,v in inner_relation.items():
        if key in v and k != v:
            return ["inner", k, key]
        if k==v:
            return ["o", k, key]
    raise Exception("not fount %d"%key)

def __conclude_list(a_list):
    no_duplicate = []
    [no_duplicate.append(i) for i in a_list if i not in no_duplicate]
    count_list = []
    for i in no_duplicate:
        count_list.append(a_list.count(i))
    return no_duplicate, count_list

def remove_duplicate(outer_relation, inner_relation, total_number, group_number, remove_mode='full'):
    if remove_mode not in ["full", "part"]:
        raise Exception("error mode %s"%remove_mode)
    for i in range(0, total_number, group_number):
        temp = []
        for j in range(i, i+group_number):
            temp.append(__search_outer_and_inner_relation(outer_relation, inner_relation, j))
        appear_list, count_list = __conclude_list([i[0:2] for i in temp])
        if remove_mode=='full':
            max_item = ksort.sort_multi_list(appear_list, count_list, by=1)[0][0]
            count = 0
            for j in temp:
                if j[0:2] != max_item:
                    if j[0] == 'outer':
                        outer_relation[j[1]].remove(j[2])
                    elif j[0] == 'inner':
                        inner_relation[j[1]].remove(j[2])
                    else:
                        inner_relation.pop(j[2])
                else:
                    if count != 0:
                        if j[0] == 'outer':
                            outer_relation[j[1]].remove(j[2])
                        elif j[0] == 'inner':
                            inner_relation[j[1]].remove(j[2])
                        else:
                            inner_relation.pop(j[2])
                    else:
                        count += 1
        else:
            pass

        

def two_layers_flat(outer_relation, inner_relation, total_number, verbose=False):
    l = 0
    for _,v in outer_relation.items():
        l+=len(v)
    if verbose:
        print("lenth: %d"%l)
    flated_outer_relation = [-2 for i in range(total_number)]
    flated_inner_relation = [-2 for i in range(total_number)]
    index = [i for i in range(total_number)]
    for k,v in outer_relation.items():
        for i in v:
            if flated_outer_relation[i] == -2:
                flated_outer_relation[i] = k
            else:
                raise Exception("wrong relation!")
    # if -1 in flated_outer_relation:
    #     raise Exception("wrong relation!")
    # for i,j in enumerate(flated_outer_relation):
    #     if i==j:
    #         flated_outer_relation[i] = -1
    for i,j in enumerate(flated_outer_relation):
        x = __map_by_dict(inner_relation, j)
        if x!=None:
            flated_inner_relation[i] = x
    # if -1 in flated_inner_relation:
    #     raise Exception("wrong relation!")
    # for i,j in enumerate(flated_inner_relation):
    #     if i==j:
    #         flated_inner_relation[i] = -1
    pop_list = []
    for i, j in enumerate(index):
        if flated_inner_relation[i] == -2 and flated_outer_relation[i] == -2:
            pop_list.append(i)
    flated_outer_relation = [j for i,j in enumerate(flated_outer_relation) if i not in pop_list]
    flated_inner_relation = [j for i,j in enumerate(flated_inner_relation) if i not in pop_list]
    index = [j for i,j in enumerate(index) if i not in pop_list]
    return {"outer":flated_outer_relation, "inner":flated_inner_relation, "index":index}, l

def __map_by_dict(dic, value):
    for k,v in dic.items():
        if value in v:
            return k
    return None


def evaluate(flated_dict, map, truth_table):
    # process pre_label
    outer_truth_label = []
    for i in flated_dict['index']:
        fd = map[i]
        outer_truth_label.append(truth_table[truth_table["FD"]==fd]["outer"].values[0])
    inner_truth_label = []
    for i in flated_dict['index']:
        fd = map[i]
        inner_truth_label.append(truth_table[truth_table["FD"]==fd]["inner"].values[0])

    cate_map, _ = __conclude_list(outer_truth_label+inner_truth_label)
    for i in range(len(outer_truth_label)):
        outer_truth_label[i] = cate_map.index(outer_truth_label[i])
    for i in range(len(inner_truth_label)):
        inner_truth_label[i] = cate_map.index(inner_truth_label[i])
    
    # adj_rand
    outer_score = metrics.adjusted_rand_score(flated_dict["outer"], outer_truth_label)
    inner_score = metrics.adjusted_rand_score(flated_dict["inner"], inner_truth_label)
    return {"outer_score": outer_score, "inner_score": inner_score}