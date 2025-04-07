import pickle

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

if __name__ == "__main__":
    ks = [1, 10, 100]
    num_pieces = [10,100,1000]
    num_sets_to_choose_from = [1000, -1]
    num_iterations = [1_000, 10_000, 100_000]
    methods = ["mh", "mh_adapt", "gibbs", "em"]
    ap_dict = {}
    for m in methods:
        ap = np.empty((len(ks), len(num_pieces), len(num_sets_to_choose_from), len(num_iterations), 2))
        for i, k in enumerate(ks):
            for j, num_p in enumerate(num_pieces):
                for e, num_sets_chosen in enumerate(num_sets_to_choose_from):
                    for p, iteration in enumerate(num_iterations):
                        for q in range(2):
                            with open(f"C:\\Users\\lyada\\Desktop\\BayesdLegoSetIdentifier\\results\\{m}-{k}-{num_p}-{num_sets_chosen}-{iteration}-10-{q}.pkl", "rb") as f:
                                data = pickle.load(f)
                            ap[i][j][e][p][q] = data['average_precision']
                        # print(ap[i][j][e][p].mean())
        ap_dict[m] = ap

        print(f"--------------{m}-------------")
        print("total sets:", ap.mean(axis=(1, 2, 3, 4)))
        print("pieces seen:", ap.mean(axis=(0, 2, 3, 4)))
        print("sets to choose from:", ap.mean(axis=(0, 1, 3, 4)))
        print("iterations:", ap.mean(axis=(0, 1, 2, 4)))

    metrics = ['Sets Placed In the Bucket', 'Pieces Seen', 'Sets to Choose From', 'Number of Iterations']
    x_labels = {
        'Sets Placed In the Bucket': ks,
        'Pieces Seen': num_pieces,
        'Sets to Choose From': ['1000', '~10,000'],
        'Number of Iterations': ['1k', '10k', '100k'],
    }
    metric_axes = {
        'Sets Placed In the Bucket': (1, 2, 3, 4),
        'Pieces Seen': (0, 2, 3, 4),
        'Sets to Choose From': (0, 1, 3, 4),
        'Number of Iterations': (0, 1, 2, 4),
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    width = 0.25  # Width of each bar

    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        tick_labels = x_labels[metric]
        x = np.arange(len(tick_labels))*1.2  # positions for each group

        for i, method in enumerate(methods):
            ap = ap_dict[method]
            means = ap.mean(axis=metric_axes[metric])  # shape: (len(tick_labels),)
            # stds = ap.std(axis=metric_axes[metric])
            print(f"{metric}, {method}", means)
            ax.bar(x + i * width, means, width, capsize=5, label=method)

        ax.set_title(f"Average Precision@10 vs {metric}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Average Precision@10")
        ax.set_xlabel(metric)
        ax.legend()
        ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    # with open("results_matrix.pkl", "wb") as f:
    #     pickle.dump(ap_dict, f)
