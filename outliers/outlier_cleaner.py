#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    for x, y, p in zip(ages, net_worths, predictions):
        cleaned_data.append((x, y, abs(y-p)))
    cleaned_data = sorted(cleaned_data, key=lambda i: i[2])
    cleaned_data = cleaned_data[:-9]

    return cleaned_data
