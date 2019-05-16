from logreg_utils import train_logreg
import numpy as np
def score_and_predict(model, X, Y):
    '''
    Given a binary classification model, predict output classification for numpy features `X`
    and evaluate accuracy against labels `Y`. Labels should be numpy array of 0s and 1s.
    Returns (accuracy, numpy array of classification probabilities)
    '''
    probs = model.predict_proba(X)[:, 1]
    clf = probs > .5
    accuracy = (np.squeeze(Y) == np.squeeze(clf)).mean()
    return accuracy, probs

def get_top_k_neuron_weights(model, k=1):
    """
    Get's the indices of the top weights based on the l1 norm contributions of the weights
    based off of https://rakeshchada.github.io/Sentiment-Neuron.html interpretation of
    https://arxiv.org/pdf/1704.01444.pdf (Radford et. al)
    Args:
        weights: numpy arraylike of shape `[d,num_classes]`
        k: integer specifying how many rows of weights to select
    Returns:
        k_indices: numpy arraylike of shape `[k]` specifying indices of the top k rows
    """
    weights = model.coef_.T
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))
    if k == 1:
        k_indices = np.array([np.argmax(weight_penalties)])
    elif k >= np.log(len(weight_penalties)):
        # runs O(nlogn)
        k_indices = np.argsort(weight_penalties)[-k:][::-1]
    else:
        # runs O(n+klogk)
        k_indices = np.argpartition(weight_penalties, -k)[-k:]
        k_indices = (k_indices[np.argsort(weight_penalties[k_indices])])[::-1]
    return k_indices

def normalize(coef):
    norm = np.linalg.norm(coef)
    coef = coef/norm
    return coef
metric='acc'
train_data=np.load('embedded_train_data.npy')
train_labels=np.load('train_labels.npy')
test_data=np.load('embedded_test_data.npy')
test_labels=np.load('test_labels.npy')

logreg_model, logreg_scores, logreg_probs, c, nnotzero = train_logreg(train_data, train_labels,test_data, test_labels, None, None,  max_iter=50,  report_metric=metric, threshold_metric=metric)
print(', '.join([str(score) for score in logreg_scores]), 'train, val, test accuracy for all neuron regression')
print(str(c)+' regularization coefficient used')
print(str(nnotzero) + ' features used in all neuron regression\n')
sentiment_neurons = get_top_k_neuron_weights(logreg_model, 5)
print('using neuron(s) %s as features for regression'%(', '.join([str(neuron) for neuron in list(sentiment_neurons.reshape(-1))])))
logreg_neuron_model, logreg_neuron_scores, logreg_neuron_probs, neuron_c, neuron_nnotzero = train_logreg(train_data, train_labels, test_data, test_labels,None, None, max_iter=50, neurons=sentiment_neurons, report_metric=metric, threshold_metric=metric)
print(', '.join([str(score) for score in logreg_neuron_scores]), 'train, val, test accuracy for all neuron regression')