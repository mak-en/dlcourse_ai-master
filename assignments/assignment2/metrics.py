def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    # print(prediction.shape, '\n')
    # print(ground_truth.shape, '\n')
    # TODO: Implement computing accuracy
    true_positives = 0
    for p in range(len(prediction)):
        if prediction[p] == ground_truth[p]:
            true_positives += 1

    # accuracy
    accuracy = (true_positives) / len(prediction)

    return accuracy
