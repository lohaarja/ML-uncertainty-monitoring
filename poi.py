def prediction_oscillation(predictions):
    switches = 0
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i-1]:
            switches += 1
    return switches / max(1, len(predictions) - 1)