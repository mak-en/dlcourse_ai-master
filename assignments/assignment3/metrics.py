def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    true_positives = 0
    for p in range(len(prediction)):
        if prediction[p] == ground_truth[p]:
            true_positives += 1

    # accuracy
    accuracy = (true_positives) / len(prediction)

    return accuracy
