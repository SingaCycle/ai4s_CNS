import dgl
import torch 
from dgl.dataloading import GraphDataLoader

dataset = dgl.data.GINDataset('MUTAG', False)
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

dataloader = GraphDataLoader(
    dataset,
    batch_size=1024,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)

for batched_graph, labels in dataloader:
    print(batched_graph)
