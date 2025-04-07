import pickle
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import poisson


matplotlib.use('TkAgg')
NUM_SETS = 1000

def load_data(file_path, num_sets=NUM_SETS):
    """Load LEGO set data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        num_sets (int): Number of sets to select.

    Returns:
        tuple: DataFrame of LEGO sets and selected set numbers.
    """
    df = pd.read_csv(file_path)
    selected_sets = df['set_number'].unique()[:num_sets]
    return df, selected_sets


def create_piece_mappings(df, selected_sets):
    """Create mappings for selected LEGO pieces.

    Args:
        df (DataFrame): LEGO set data.
        selected_sets (array): Selected set numbers.

    Returns:
        tuple: Array of selected pieces and dictionary mapping piece numbers to indices.
    """
    selected_pieces = df[df['set_number'].isin(selected_sets)]['piece_part_num'].unique()
    piece_to_idx = {piece: idx for idx, piece in enumerate(selected_pieces)}
    return selected_pieces, piece_to_idx


def build_set_piece_matrix(df, selected_sets, piece_to_idx):
    """Build matrix of piece probabilities for each set.

    Args:
        df (DataFrame): LEGO set data.
        selected_sets (array): Selected set numbers.
        piece_to_idx (dict): Mapping of pieces to indices.

    Returns:
        ndarray: Matrix of piece probabilities.
    """
    #MxN
    set_piece_counts = np.zeros((len(selected_sets), len(piece_to_idx)))

    for i, set_num in enumerate(selected_sets):
        set_df = df[df['set_number'] == set_num]
        for _, row in set_df.iterrows():
            piece_idx = piece_to_idx[row['piece_part_num']]
            set_piece_counts[i, piece_idx] += row['quantity_of_piece']

    epsilon = 1e-6
    set_piece_prob_matrix = set_piece_counts + epsilon
    set_piece_prob_matrix /= set_piece_prob_matrix.sum(axis=1, keepdims=True)

    return set_piece_prob_matrix


def sample_lego_pieces(random_sets, selected_sets, set_piece_prob_matrix, num_pieces=100):
    """Sample LEGO pieces based on random sets and piece probabilities.

    Args:
        random_sets (array): Randomly selected sets.
        selected_sets (array): All selected sets.
        set_piece_prob_matrix (ndarray): Piece probability matrix.
        num_pieces (int): Number of pieces to sample.

    Returns:
        list: List of sampled LEGO piece indices.
    """
    lego_pieces_seen = []

    for _ in range(num_pieces):
        chosen_set = np.random.choice(random_sets)
        chosen_set_idx = np.where(selected_sets == chosen_set)[0][0]
        chosen_piece_idx = np.random.choice(
            len(set_piece_prob_matrix[chosen_set_idx]),
            p=set_piece_prob_matrix[chosen_set_idx]
        )
        lego_pieces_seen.append(chosen_piece_idx)

    return lego_pieces_seen

def expectation_maximization(lego_pieces_seen, set_piece_prob_matrix, epsilon=1e-4, max_iter=100000):
    num_sets, num_piece_types = set_piece_prob_matrix.shape
    pis = np.ones(num_sets) / num_sets  # TODO: maybe there are better ways to initialize?

    # pre-index the piece probability matrix
    piece_indices = np.array(lego_pieces_seen)
    likelihoods = set_piece_prob_matrix[:, piece_indices]  # shape: (num_sets, num_observations)

    for _ in range(max_iter):
        # e-step: compute responsibilities γ_ij = P(S_j | piece_i)
        weighted_likelihoods = pis[:, None] * likelihoods  # shape: (num_sets, num_observations)
        responsibilities = weighted_likelihoods / (weighted_likelihoods.sum(axis=0, keepdims=True) + 1e-12)

        # m-step: update pis based on average responsibilities per set
        new_pis = responsibilities.sum(axis=1) / responsibilities.shape[1]

        if np.all(np.abs(new_pis - pis) <= epsilon):
            break

        pis = new_pis

    return pis
def gibbs_sampling(lego_pieces_seen, set_piece_prob_matrix, num_sets, alpha=1, iterations=1000):
    num_pieces = len(lego_pieces_seen)
    z = np.random.choice(num_sets, size=num_pieces)
    pis = np.random.dirichlet(np.ones(num_sets)*alpha)

    for _ in tqdm(range(iterations)):
        # Update assignments (z)
        for i, piece_idx in enumerate(lego_pieces_seen):
            probs = pis*set_piece_prob_matrix[:, piece_idx]
            probs /= probs.sum()
            # Assigns a set to a piece based on probs
            z[i] = np.random.choice(num_sets, p=probs)

        # Update pis
        set_count = np.bincount(z, minlength=num_sets)
        pis = np.random.dirichlet(alpha+set_count)
    return pis, z


def calculate_log_likelihood(lego_pieces_seen, set_piece_prob_matrix, theta):
    """
    Calculates log-likelihood of observing the list of pieces (by ID),
    assuming each piece is drawn independently from the set mixture.

    - lego_pieces_seen: list of length R (observed piece IDs, values 0 to N-1)
    - set_piece_prob_matrix: shape (M, N), probabilities of each part in each set
    - theta: shape (M,), mixture weights over the M sets
    """
    # mixture distribution over parts
    mixture_over_parts = set_piece_prob_matrix.T@theta  # shape: (N,)
    mixture_over_parts = np.clip(mixture_over_parts, 1e-12, None)  # avoid log(0)

    # for each observed piece, get its probability under the mixture
    log_probs = np.log(mixture_over_parts[lego_pieces_seen])

    return np.sum(log_probs)

def calculate_log_prior(theta, lam=1.0):
    """
    Log-prior: exponential prior over theta (sparse prior)
    p(theta) = exp(-lam * theta)
    """
    return -lam * np.sum(theta)


def metropolis_hastings(lego_pieces_seen, set_piece_prob_matrix, num_sets, adapt, iterations=100000, eps=0.5):
    # calculate likelihood and prior
    theta = np.random.rand(num_sets)
    theta = np.clip(theta, 0, None)

    log_likelihood = calculate_log_likelihood(lego_pieces_seen, set_piece_prob_matrix, theta)
    log_prior = calculate_log_prior(theta)
    log_posterior = log_likelihood + log_prior

    samples = []

    sum_accept_ratios, len_accept_ratios = 0,0

    for i in tqdm(range(iterations)):
        if adapt and i % 100 == 0 and i > 0:
            if sum_accept_ratios/len_accept_ratios < 0.2:
                eps *= 0.9  # reduce step size, increase acceptance rate
            elif sum_accept_ratios/len_accept_ratios > 0.5:
                eps *= 1.1  # increase step size, reduce acceptance rate

        sample_theta = theta + np.random.normal(0, eps, size=theta.shape)
        sample_theta = np.clip(sample_theta, 0, None)

        proposed_log_likelihood = calculate_log_likelihood(lego_pieces_seen, set_piece_prob_matrix, sample_theta)
        proposed_log_prior = calculate_log_prior(sample_theta)
        proposed_log_posterior = proposed_log_likelihood + proposed_log_prior

        log_accept_ratio = proposed_log_posterior - log_posterior
        accept_ratio = np.exp(min(0, log_accept_ratio))
        sum_accept_ratios += accept_ratio
        len_accept_ratios += 1

        if np.random.uniform() < accept_ratio:
            theta = sample_theta
            log_posterior = proposed_log_posterior

        samples.append(theta.copy())

    samples = np.array(samples)

    # Normalize each sample to get set probabilities
    probs = samples / (samples.sum(axis=1, keepdims=True) + 1e-10)
    print(f"accept ratio: {sum_accept_ratios/len_accept_ratios}")
    map_probs = probs.mean(axis=0)
    return map_probs





# TODO: Evaluate on different num_pieces_seen, num_sets, and speed
def evaluate(pis, selected_sets, random_sets, num_sets_to_eval, return_curves=False):
    """Evaluate the model by computing precision, recall, F1, and average precision (AP) at each k.

    Args:
        pis (ndarray): Mixture weights.
        selected_sets (array): All selected sets.
        random_sets (array): Randomly selected sets.
        num_sets_to_eval (int): Max number of sets to evaluate.
        return_curves (bool): Whether to return full precision/recall/F1 curves.

    Returns:
        dict: final recall, precision, f1, AP, and optionally the full curves.
    """
    top_indices = np.argsort(pis)[::-1][:num_sets_to_eval]
    top_sets = selected_sets[top_indices]

    actual_set = set(random_sets)

    precision_curve = []
    recall_curve = []
    f1_curve = []

    sum_precisions = 0

    for k in range(1, num_sets_to_eval + 1):
        current_pred = top_sets[:k]
        tp = len(set(current_pred) & actual_set)
        recall = tp / len(actual_set) if actual_set else 0
        precision = tp / k
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        # For AP: accumulate precision only at points where a correct set is predicted
        if top_sets[k - 1] in actual_set:
            sum_precisions += precision

        precision_curve.append(precision)
        recall_curve.append(recall)
        f1_curve.append(f1)

    final_recall = recall_curve[-1]
    final_precision = precision_curve[-1]
    final_f1 = f1_curve[-1]
    ap = sum_precisions / len(actual_set) if actual_set else 0

    print("Final mixture weights:", pis)
    print("Actual sets in the random bucket:", random_sets)
    print(f"Final Recall@{num_sets_to_eval}: {final_recall:.4f}")
    print(f"Final Precision@{num_sets_to_eval}: {final_precision:.4f}")
    print(f"Final F1@{num_sets_to_eval}: {final_f1:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print("Predicted top-k sets based on pis:", top_sets)

    result = {
        "recall": final_recall,
        "precision": final_precision,
        "f1": final_f1,
        "average_precision": ap,
    }

    if return_curves:
        result.update({
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "f1_curve": f1_curve,
        })

    return result




def main(method, time, k=3, num_pieces=14, num_sets_to_choose=-1, num_iterations=100000, num_sets_to_eval=10, return_curves=True):
    """Main execution function to load data, train the model, and evaluate recall."""
    df, selected_sets = load_data('lego_sets_pieces.csv')
    selected_sets = selected_sets[:num_sets_to_choose]
    selected_pieces, piece_to_idx = create_piece_mappings(df, selected_sets)
    set_piece_prob_matrix = build_set_piece_matrix(df, selected_sets, piece_to_idx)

    # Tuneable: method to use, k (number of sets in the bucket), selected_sets (how many sets to choose from), num_pieces (the number of burn in lego pieces), num_iterations

    random_sets = np.random.choice(selected_sets, size=k, replace=False)
    lego_pieces_seen = sample_lego_pieces(random_sets, selected_sets, set_piece_prob_matrix, num_pieces=num_pieces)

    if method == "em":
        # This doesn't depend on number iterations. Stops when difference is less than epsilon
        pis = expectation_maximization(lego_pieces_seen, set_piece_prob_matrix, max_iter=num_iterations)
    elif method == "gibbs":
        pis, z = gibbs_sampling(lego_pieces_seen, set_piece_prob_matrix, len(selected_sets), iterations=num_iterations)
    else:
        if method == "mh":
            adapt = False
        else:
            adapt = True
        pis = metropolis_hastings(lego_pieces_seen, set_piece_prob_matrix, len(selected_sets), adapt=adapt, iterations=num_iterations)
    result = evaluate(pis, selected_sets, random_sets, num_sets_to_eval, return_curves=return_curves)
    if return_curves:
        plt.plot(range(1, 11), result["precision_curve"], label="Precision@k")
        plt.plot(range(1, 11), result["recall_curve"], label="Recall@k")
        plt.plot(range(1, 11), result["f1_curve"], label="F1@k")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.title(f"Evaluation Metrics at Different k for {method}")
        plt.legend()
        filename = f"plots/plot_k={k}_pieces={num_pieces}_setseval={num_sets_to_eval}_sets={num_sets_to_choose}_iters={num_iterations}_method={method}_time={time}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
    with open(f"./results/{method}-{k}-{num_pieces}-{num_sets_to_choose}-{num_iterations}-{num_sets_to_eval}-{time}.pkl", "wb") as f:
        pickle.dump(result, f)
    return result['recall']


# todo:
# - do evaluations for the four methods (EM, gibbs, MH, dynamic MH)
# 	- how it varies with burn in lego pieces
# 	- how it varies with number of sets
# 	- how it varies with number of selected sets
# 	- how it varies with number of iterations
# - write the theory of the three methods
# - write experiment methodology
# - analyze the results of the experiments
# - write what could be done for the future work
# 	- Actually todo this!: Change prior to use data on how many sets of this type were produced
# 	- Combinations of the methods (ie. use one method for first part and then use another methods to get finer details?)
# 	- Other MCMC methods (HMC for example)
#   - Open question: how to evaluate, does F1 score make sense, do we maybe care about recall more?

if __name__ == "__main__":
    ks = [1, 10, 100]
    nums_pieces = [10, 100, 1000]
    nums_sets_to_choose = [1000, -1]
    nums_iterations = [1000, 10000, 100000]
    methods = ["em"]
    # TODO: Do with EM later with different params (eg. num_iterations doesn't matter)

    grid = list(product(ks, nums_pieces, nums_sets_to_choose, nums_iterations, methods))

    for k, num_pieces, num_sets_to_choose, num_iterations, method in tqdm(grid):
        [main(method=method, time=i, num_sets_to_choose=num_sets_to_choose, num_pieces=num_pieces, k=k, num_iterations=num_iterations) for i in tqdm(range(2))]
