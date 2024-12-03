#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        eta = 1
        # Sign function.
        y_hat_i = np.argmax(self.W.dot(x_i))  # Predicted label
        if y_hat_i != y_i:
            # Perceptron update.
            self.W[y_i, :] += eta * x_i
            self.W[y_hat_i, :] -= eta * x_i

        # raise NotImplementedError # Q1.1 (a)


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_i_one_hot = np.zeros((np.size(self.W, 0),1))
        y_i_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is num_labels x num_features.
        if l2_penalty:
            self.W = (1 - learning_rate * l2_penalty) * self.W + learning_rate * (y_i_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)
        else:
            self.W = self.W + learning_rate * (y_i_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)

        # raise NotImplementedError # Q1.2 (a,b)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.W1 = np.random.normal(0.1, 0.1, size=(self.hidden_size, self.n_features))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.normal(0.1, 0.1, size=(self.n_classes, self.hidden_size))
        self.b2 = np.zeros(self.n_classes)

        # raise NotImplementedError # Q1.3 (a)

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, output):
        return np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        predicted_labels = []
        for x in X:
            # Comoute forward pass
            h0 = x
            z1 = self.W1.dot(h0) + self.b1
            h1 = self.relu(z1)
            z2 = self.W2.dot(h1) + self.b2
            h2 = self.softmax(z2)
            # Get the class with the highest probability
            y_hat = np.argmax(h2)
            predicted_labels.append(y_hat)

        return np.array(predicted_labels)
        # raise NotImplementedError # Q1.3 (a)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        total_loss = 0
        one_hot = np.zeros((np.size(y, 0), self.n_classes))
        for i in range(np.size(y, 0)):
            one_hot[i, y[i]] = 1
        y_train_ohe = one_hot

        # For each observation and target
        for x_i, y_i in zip(X, y_train_ohe):
            # Comoute forward pass
            h0 = x_i
            z1 = self.W1.dot(h0) + self.b1
            h1 = self.relu(z1)
            z2 = self.W2.dot(h1) + self.b2
            probs = self.softmax(z2)
            
            # Compute Loss and Update total loss
            loss = -y_i.dot(np.log(probs + 1e-8))            
            total_loss+=loss

            # Compute backpropagation
            grad_z2 = probs - y_i  # Grad of loss wrt p
            # Gradient of hidden parameters.
            grad_W2 = grad_z2[:, None].dot(h1[:, None].T)
            grad_b2 = grad_z2
            # Gradient of hidden layer below.
            grad_h1 = self.W2.T.dot(grad_z2)
            # Gradient of hidden layer below before activation.
            grad_z1 = grad_h1 * (z1 > 0)
            # Gradient of hidden parameters.
            grad_W1 = grad_z1[:, None].dot(h0[:, None].T)
            grad_b1 = grad_z1

            # Gradient updates.
            self.W1 -= learning_rate*grad_W1
            self.b1 -= learning_rate*grad_b1
            self.W2 -= learning_rate*grad_W2
            self.b2 -= learning_rate*grad_b2

        return total_loss
        # raise NotImplementedError # Q1.3 (a)


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # # plot
    # plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    # if opt.model == 'mlp':
    #     plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    # elif opt.model == 'logistic_regression':
    #     plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    # with open(f"Q1-{opt.model}-results.txt", "w") as f:
    #     f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
    #     f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")

    ########################################
    # I CHANGED THE PATHS - REVERT TO SUBMIT
    ########################################
    path_to_directory = "/home/morais/deep_learning_project"
    plot(epochs, train_accs, valid_accs, filename=f"{path_to_directory}/images/Q1-{opt.model}-accs")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"{path_to_directory}/images/Q1-{opt.model}-loss")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"{path_to_directory}/images/Q1-{opt.model}-w_norms")
    with open(f"{path_to_directory}/results/Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
