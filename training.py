import numpy as np
from functions import softmax, cross_entropy_loss
def train(X_train, y_train, learning_rate=0.01, num_epochs=100):
    num_samples, height, width, channels = X_train.shape
    num_classes = 3

    W = np.random.randn(height * width * channels, num_classes)
    b = np.zeros(num_classes)

    X_train_flat = X_train.reshape(num_samples, -1)

    for epoch in range(num_epochs):
        logits = np.dot(X_train_flat, W) + b
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_train)

        grad_logits = probs.copy()
        grad_logits[np.arange(num_samples), y_train] -= 1
        grad_logits /= num_samples

        grad_W = np.dot(X_train_flat.T, grad_logits)
        grad_b = np.sum(grad_logits, axis=0)
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W, b

def predict(X, W, b):
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    logits = np.dot(X_flat, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)