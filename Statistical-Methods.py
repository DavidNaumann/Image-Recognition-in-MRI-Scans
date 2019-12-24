def statistical_error(actual, predicted):
    error = ((abs(predicted - actual))/actual)
    return error