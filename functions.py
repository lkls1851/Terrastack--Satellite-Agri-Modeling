import numpy as np
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    num_samples = len(y)
    correct_log_probs = -np.log(probs[np.arange(num_samples), y])
    loss = np.sum(correct_log_probs) / num_samples
    return loss