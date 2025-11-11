import numpy as np

def precision(y_true, y_pred):
    """
    Calculates precision.
    
    Args:
        y_true (list): A list of true labels (1 for relevant, 0 for non-relevant).
        y_pred (list): A list of predicted labels (1 for relevant, 0 for non-relevant).
        
    Returns:
        float: The precision score.
    """
    true_positives = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    predicted_positives = np.sum(y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true, y_pred):
    """
    Calculates recall.
    
    Args:
        y_true (list): A list of true labels (1 for relevant, 0 for non-relevant).
        y_pred (list): A list of predicted labels (1 for relevant, 0 for non-relevant).
        
    Returns:
        float: The recall score.
    """
    true_positives = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    actual_positives = np.sum(y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred):
    """
    Calculates the F1-score.
    
    Args:
        y_true (list): A list of true labels (1 for relevant, 0 for non-relevant).
        y_pred (list): A list of predicted labels (1 for relevant, 0 for non-relevant).
        
    Returns:
        float: The F1-score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def mean_average_precision(y_true, y_pred_scores):
    """
    Calculates Mean Average Precision (MAP).
    
    Args:
        y_true (list): A list of true labels (1 for relevant, 0 for non-relevant).
        y_pred_scores (list): A list of predicted scores.
        
    Returns:
        float: The MAP score.
    """
    y_true = np.array(y_true)
    y_pred_scores = np.array(y_pred_scores)
    
    sorted_indices = np.argsort(y_pred_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    average_precisions = []
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            prec_at_k = np.sum(y_true_sorted[:i+1]) / (i + 1)
            average_precisions.append(prec_at_k)
            
    return np.mean(average_precisions) if average_precisions else 0

def mean_reciprocal_rank(y_true, y_pred_scores):
    """
    Calculates Mean Reciprocal Rank (MRR).
    
    Args:
        y_true (list): A list of true labels (1 for relevant, 0 for non-relevant).
        y_pred_scores (list): A list of predicted scores.
        
    Returns:
        float: The MRR score.
    """
    y_true = np.array(y_true)
    y_pred_scores = np.array(y_pred_scores)
    
    sorted_indices = np.argsort(y_pred_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    for i, label in enumerate(y_true_sorted):
        if label == 1:
            return 1 / (i + 1)
            
    return 0

def normalized_discounted_cumulative_gain(y_true, y_pred_scores, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        y_true (list): A list of true relevance scores (can be graded, e.g., 0-5).
        y_pred_scores (list): A list of predicted scores.
        k (int): The number of top results to consider.
        
    Returns:
        float: The NDCG score.
    """
    y_true = np.array(y_true)
    y_pred_scores = np.array(y_pred_scores)
    
    # DCG
    sorted_indices = np.argsort(y_pred_scores)[::-1]
    y_true_sorted = y_true[sorted_indices][:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(y_true_sorted / discounts)
    
    # IDCG
    ideal_sorted_indices = np.argsort(y_true)[::-1]
    ideal_y_true_sorted = y_true[ideal_sorted_indices][:k]
    idcg = np.sum(ideal_y_true_sorted / discounts)
    
    return dcg / idcg if idcg > 0 else 0
