from numpy import nan


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    # finding all trues (true positives + false positives) in prediction:
    all_positives = 0
    for p in prediction:
        if p == True:
            all_positives += 1

    # finding true positives
    true_positives = 0
    for p in range(len(prediction)):
        if prediction[p] == ground_truth[p] and prediction[p] == True:
            true_positives += 1

    # precision
    if true_positives and all_positives == 0:
        precision = float('inf')
    elif true_positives == 0 and all_positives == 0:
        precision = float('nan')
    else:
        precision = true_positives / all_positives

    # finding all trues (true positives + false negatives in prediction)
    # in ground_truth:
    all_true_positives = 0
    for p in ground_truth:
        if p == True:
            all_true_positives += 1

    # recall
    recall = true_positives / all_true_positives

    # f1
    if precision == 0 and recall == 0:
        f1 = float('nan')
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))

    #finding true negatives
    true_negatives = 0
    for p in range(len(prediction)):
        if prediction[p] == ground_truth[p] and prediction[p] == False:
            true_negatives += 1
    
    # accuracy
    accuracy = (true_positives + true_negatives) / len(prediction)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    true_positives = 0
    for p in range(len(prediction)):
        if prediction[p] == ground_truth[p]:
            true_positives += 1

    # accuracy
    accuracy = (true_positives) / len(prediction)

    return accuracy
