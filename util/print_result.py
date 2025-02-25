import os
os.system('clear')
import numpy as np

def print_top_k_accuracy(results, k_list=[1, 5]):
    """
    Computes the top-k accuracy for the result indices.
    
    Args:
        results (list): A list of dictionaries containing list of pairs: [true_indicies, retrieved_indicies].
    Returns:
        acc_str: A string show top k accuracy
    """
    top_k_acc = {k: [] for k in k_list}

    for meta in results:
        true_indicies = meta[0]
        retrieved_indicies = meta[1]  # Predicted ranked lists

        for k in k_list:
            acc = [true_i in pred[:k] for true_i, pred in zip(true_indicies, retrieved_indicies)]
            top_k_acc[k] += acc

    # Compute final average accuracy for each k
    top_k_acc = {k: np.mean(v) for k, v in top_k_acc.items()}

    acc_str = ', '.join([f'top-{k}-acc: {100 * top_k_acc[k]:.2f} %' for k in k_list])
    return acc_str
