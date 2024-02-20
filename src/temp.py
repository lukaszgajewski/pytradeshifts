import networkx as nx
import numpy as np
from itertools import chain
import pandas as pd

G = nx.Graph()
G.add_edge("u", "q", weight=1.5)
G.add_edge("q", "r", weight=4)
G.add_edge("r", "v", weight=2)
G.add_edge("q", "v", weight=1)
G.add_edge("x", "y", weight=10)
W = nx.to_pandas_adjacency(G)
p1 = dict(
    nx.all_pairs_dijkstra(
        G,
        weight=lambda _, __, attr: (
            attr["weight"] ** -1 if attr["weight"] != 0 else np.inf
        ),
    )
)
d = pd.DataFrame(0.0, index=G.nodes(), columns=G.nodes())
fi = pd.DataFrame(0.0, index=G.nodes(), columns=G.nodes())
for source, (dists, paths) in p1.items():
    for target, cost in dists.items():
        d.loc[source, target] = cost
    for target, path in paths.items():
        fi.loc[source, target] = nx.path_weight(G, path, weight="weight")
print(W)
print(fi)
W_ideal = (W + fi) / 2
W_ideal = fi
print(W_ideal)
E_ideal = np.sum(W_ideal.values)
E = np.sum((1 / d).values[d != 0])
print(E, E_ideal, E / E_ideal)
