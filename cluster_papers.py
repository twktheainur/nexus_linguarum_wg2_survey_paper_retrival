from argparse import ArgumentParser
import itertools
import json
import pathlib
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

import matplotlib.pyplot as plt


from tqdm import tqdm, trange
import numpy as np


parser = ArgumentParser(description="Embed paper titles and abstracts.")

parser.add_argument(
    "input_papers",
    type=pathlib.Path,
    help="Input json file resulting from SemanticScholar extraction")

parser.add_argument(
    "input_embeddings",
    type=pathlib.Path,
    help="Input npy embedding file")

parser.add_argument(
    "input_scores",
    type=pathlib.Path,
    help="Input npy with paper scores file")

parser.add_argument("--dim_red",
                    "-dr",
                    nargs=1,
                    default=["isomap"],
                    choices=["mds", "lle", "isomap", "t-sne"],
                    help="The dimension reduction technique to use. Default: isomap.")

parser.add_argument("--num_dim",
                    "-nd",
                    nargs=1,
                    type=int,
                    default=[3],
                    help="The number of dimensions. Default: 3")

parser.add_argument("--clustering",
                    "-c",
                    nargs=1,
                    default=["agglo"],
                    choices=["kmeans", "agglo", "spectral", "dbscan"],
                    help="The clustering method to use. Default: agglo")

parser.add_argument("--num_clusters", "-nc", nargs=1, type=int, default=[5])


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.input_papers) as f:
        papers = json.load(f)

    embedding = np.load(args.input_embeddings)

    scores = np.load(args.input_scores)

    connectivity = kneighbors_graph(
        embedding, n_neighbors=50, include_self=False
    )
    # make connectivity symmetric
    # connectivity = 0.5 * (connectivity + connectivity.T)

    clustering = AgglomerativeClustering(
        n_clusters=args.num_clusters[0], linkage="ward", connectivity=connectivity
    ).fit(embedding)

    print(np.unique(clustering.labels_, return_counts=True))

    from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE, MDS

    n_components = args.num_dim[0]
    if args.dim_red[0] == "isomap":
        red = Isomap(n_components=n_components)
    elif args.dim_red[0] == "t-sne":
        red = TSNE(n_components=n_components)
    elif args.dim_red[0] == "lle":
        red = LocallyLinearEmbedding(n_components=n_components)
    else:
        red = MDS(n_components=n_components)

    matrix_2D = red.fit_transform(embedding)

    pairs = itertools.product(range(n_components), repeat=2)
    for pair in pairs:
        if pair[0] != pair[1]:
            fig = plt.figure(dpi=300)
            plt.scatter(matrix_2D.T[pair[0]],
                        matrix_2D.T[pair[1]], c=clustering.labels_)
            plt.colorbar()
            fig.savefig(f"projection_{pair[0]}_{pair[1]}.png")

    clustering.labels_.tolist()
    cluster_lists = []
    for cli in range(0, args.num_clusters[0]):
        cluster_list = []
        for docindex in np.argwhere(clustering.labels_ == cli).tolist():
            paper = papers[docindex[0]]
            cluster_list.append(paper)
        print("Cluster", cli, len(cluster_list))
        cluster_lists.append(cluster_list)
        with open(f"cluster_{cli}_papers.json", "w") as f:
            json.dump(cluster_list, f)
