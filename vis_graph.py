import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

index = 0
n_predicate=20

def rev_yaxis(coord_list):
    ylist = []
    for x, y in coord_list:
        ylist.append(y)
    ymax = max(ylist)
    newlist = []
    for x, y in coord_list:
        newlist.append((x, ymax - y))
    return newlist

# get label-code information
with open("datasets/vg_bm/VG-SGG-dicts.json") as f:
    d = json.load(f)
label2idx = d["label_to_idx"]
predicate2idx = d["predicate_to_idx"]
idx2label = sorted(label2idx, key=lambda k:label2idx[k])

# get bbox information
prediction = torch.load("results/predictions.pth")
fields = prediction[index].extra_fields
label_len = fields["labels"].size(0)
coord_list = list((int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in prediction[index].bbox)
coord_list = rev_yaxis(coord_list)
bbox_coord = dict(zip(np.arange(0, label_len).tolist(), coord_list))
labeldict = dict(zip(np.arange(0, label_len).tolist(), fields["labels"].tolist()))
print(bbox_coord)
print(labeldict)    #dict of bboxidx -> classidx

# get bbox pair information
pre_pre = torch.load("results/predictions_pred.pth")
fields = pre_pre[index].extra_fields
vals = []
inds = []
for scores in fields["scores"]:
    val, ind = scores.max(0)
    vals.append(float(val))
    inds.append(int(ind))
vals = np.array(vals)        #list of relness of pairs
pairs = fields["idx_pairs"]  #pairs of bboxidx
top_idxs = []
top_pairs = []

graph = nx.Graph()
bbox_included = []
for i in range(n_predicate):
    idx = np.where(vals==np.sort(vals)[-i])[0][0] #idx of ith top relness pair
    top_idxs.append(int(idx))
    top_pairs.append(pairs[int(idx)])
    bbox_included += pairs[int(idx)].tolist()
    graph.add_edge(int(pairs[int(idx)][0]), int(pairs[int(idx)][1]))
print(bbox_included)
graphdict = {}
posdict = {}
for idx in range(0, len(labeldict)):
    if idx in bbox_included:
        graphdict.update([(idx, idx2label[labeldict[idx]-1])])
        posdict.update([(idx, bbox_coord[idx])])
print(graphdict)
print(posdict)

nx.draw_networkx(graph, pos=posdict, labels=graphdict)
plt.savefig("results/sgraph_idx{}_.png".format(index))

