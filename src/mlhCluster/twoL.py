from Kkit import ksort
import numpy as np
from sklearn import metrics as m1
import scipy.cluster.hierarchy as sch
from kneed import KneeLocator
# from yellowbrick.utils import KneeLocator
import warnings
import matplotlib.pyplot as plt
from . import k_cluster
from . import metrics as m2

def cluster(linkage, outer_distance_threshold, inner_distance_threshold):
    if outer_distance_threshold >= inner_distance_threshold:
        raise Exception("outer_distance_threshold must small than inner_distance_threshold")
    _, Node_list = k_cluster.to_tree(linkage)
    k_cluster.compress_tree(Node_list, outer_distance_threshold)
    outer_relation = k_cluster.gen_classes(Node_list,outer_distance_threshold)
    k_cluster.clean_tree(Node_list,outer_distance_threshold)
    k_cluster.compress_tree(Node_list, inner_distance_threshold)
    inner_relation = k_cluster.gen_classes(Node_list, inner_distance_threshold)  
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
    for i in a_list:
        if i not in no_duplicate:
            no_duplicate.append(i)
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
        appear_list, count_list = __conclude_list([relation[0:2] for relation in temp])
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

        

def flat(outer_relation, inner_relation, total_number, verbose=False):
    l = 0
    for _,v in outer_relation.items():
        l+=len(v)
    if verbose:
        print("length: %d"%l)
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


def adj_rand_index(flated_dict, token_map, truth_table):
    # process pre_label
    outer_truth_label = []
    for i in flated_dict['index']:
        fd = token_map[i]
        outer_truth_label.append(truth_table[truth_table["FD"]==fd]["outer"].values[0])
    inner_truth_label = []
    for i in flated_dict['index']:
        fd = token_map[i]
        inner_truth_label.append(truth_table[truth_table["FD"]==fd]["inner"].values[0])

    cate_map, _ = __conclude_list(outer_truth_label+inner_truth_label)
    for i in range(len(outer_truth_label)):
        outer_truth_label[i] = cate_map.index(outer_truth_label[i])
    for i in range(len(inner_truth_label)):
        inner_truth_label[i] = cate_map.index(inner_truth_label[i])
    
    # adj_rand
    outer_score = m1.adjusted_rand_score(flated_dict["outer"], outer_truth_label)
    inner_score = m1.adjusted_rand_score(flated_dict["inner"], inner_truth_label)
    return {"outer_score": outer_score, "inner_score": inner_score}

def outer_silhouette_index(flated_dict, vecs, distance_metric):
    outer_vecs = vecs[flated_dict["index"]]
    return m1.silhouette_score(outer_vecs, flated_dict["outer"], metric=distance_metric)

def one_step_determine_distance(vecs, linkage_metric="cosine", linkage_method="average", start=0, end=2, step=0.001, figsize=(10,16), img_path=None, dpi=600, online=False):
    fig = plt.figure(figsize=figsize)
    linkage_matrix = sch.linkage(vecs, metric=linkage_metric, method=linkage_method)
    shape_of_linkage = linkage_matrix.shape
    # determine outer layer
    outer_num = []
    outer_distance_threshold_list = []
    outer_silhouette_score = []
    outer_distortion_score = []
    for i in np.arange(start,end,step):
        outer_relation, inner_relation = cluster(linkage_matrix, i, 2)
        num_of_clusters = len(outer_relation)
        if 2<=num_of_clusters<=shape_of_linkage[0]:
            flated_dict, _ = flat(outer_relation, inner_relation, shape_of_linkage[0]+1, verbose=False)
            if num_of_clusters not in outer_num:
                outer_num.append(num_of_clusters)
                outer_distortion_score.append(m2.distortion_score(vecs, flated_dict["outer"], linkage_metric))
                outer_silhouette_score.append(m1.silhouette_score(vecs, flated_dict["outer"], metric=linkage_metric))
                outer_distance_threshold_list.append(i)
    outer_num.reverse()
    outer_distance_threshold_list.reverse()
    outer_silhouette_score.reverse()
    outer_distortion_score.reverse()
    elbow_locator = KneeLocator(outer_num, outer_distortion_score, curve="convex", direction="decreasing", online=online)
    # elbow_locator = KneeLocator(outer_num, outer_distortion_score, curve_nature="convex", curve_direction="decreasing")
    if elbow_locator.knee is None:
        elbow_value_1 = None
        elbow_score_1 = 0
        outer_distance = 1
        warning_message = (
            "No 'knee' or 'elbow' point detected, "
            "pass `locate_elbow=False` to remove the warning"
        )
        warnings.warn(warning_message)
    else:
        elbow_value_1 = elbow_locator.knee
        elbow_score_1 = outer_distortion_score[outer_num.index(elbow_value_1)]
        outer_distance = outer_distance_threshold_list[outer_num.index(elbow_value_1)]
    fig.add_subplot(211)
    plt.plot(outer_num, outer_distortion_score, color="black", marker="o", linewidth=2, label="sum of squared distances")
    plt.axvline(elbow_value_1, color="red", linewidth=2, label="elbow at k=%d, score=%.3f"%(elbow_value_1, elbow_score_1), linestyle="--")
    plt.legend(loc="best")
    plt.xlabel("k")
    plt.ylabel("distortion")
    plt.title("Outer layer evaluation")
    # determine inner layer
    inner_num = []
    inner_distance_threshold_list = []
    inner_silhouette_score = []
    inner_distortion_score = []
    for i in np.arange(outer_distance+step,end,step):
        outer_relation, inner_relation = cluster(linkage_matrix, outer_distance, i)
        num_of_clusters = len(inner_relation)
        if 2<=num_of_clusters<=shape_of_linkage[0]:
            flated_dict, _ = flat(outer_relation, inner_relation, shape_of_linkage[0]+1, verbose=False)
            if num_of_clusters not in inner_num:
                inner_num.append(num_of_clusters)
                inner_distortion_score.append(m2.distortion_score(vecs, flated_dict["inner"], linkage_metric))
                inner_silhouette_score.append(m1.silhouette_score(vecs, flated_dict["inner"], metric=linkage_metric))
                inner_distance_threshold_list.append(i)
    inner_num.reverse()
    inner_distance_threshold_list.reverse()
    inner_silhouette_score.reverse()
    inner_distortion_score.reverse()
    elbow_locator = KneeLocator(inner_num, inner_distortion_score, curve="convex", direction="decreasing", online=online)
    # elbow_locator = KneeLocator(inner_num, inner_distortion_score, curve_nature="convex", curve_direction="decreasing")
    if elbow_locator.knee is None:
        elbow_value_2 = None
        elbow_score_2 = 0
        inner_distance = 1
        warning_message = (
            "No 'knee' or 'elbow' point detected, "
            "pass `locate_elbow=False` to remove the warning"
        )
        warnings.warn(warning_message)
    else:
        elbow_value_2 = elbow_locator.knee
        elbow_score_2 = inner_distortion_score[inner_num.index(elbow_value_2)]
        inner_distance = inner_distance_threshold_list[inner_num.index(elbow_value_2)]
    fig.add_subplot(212)
    plt.plot(inner_num, inner_distortion_score, color="black", marker="o", linewidth=2, label="sum of squared distances")
    plt.axvline(elbow_value_2, color="red", linewidth=2, label="elbow at k=%d, score=%.3f"%(elbow_value_2, elbow_score_2), linestyle="--")
    plt.legend(loc="best")
    plt.xlabel("k")
    plt.ylabel("distortion")
    plt.title("Inner layer evaluation")
    plt.show()
    if img_path!=None:
        fig.savefig(img_path, dpi=dpi)
    return outer_distance, inner_distance, {"cluster_num": outer_num, "distance": outer_distance_threshold_list, "silhouette": outer_silhouette_score, "distortion": outer_distortion_score}, {"cluster_num": inner_num, "distance": inner_distance_threshold_list, "silhouette": inner_silhouette_score, "distortion": inner_distortion_score}
