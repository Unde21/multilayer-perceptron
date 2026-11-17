"""Basic statistics computed without numpy or pandas."""


def ft_count(values) -> float:
    """Return the number of values.

    `values` must be a list without NaN.
    """
    return float(sum(1 for val in values))


def ft_mean(values, count) -> float:
    """Return the mean of the values, or NaN if there is none.

    `values` must be a list without NaN, `count` its length.
    """
    if count == 0:
        return float('nan')
    total_sum = sum(val for val in values)
    return total_sum / count


def ft_std(values, mean, count) -> float:
    """Return the sample standard deviation of the values (divided by
    count - 1), or NaN if there are fewer than 2 values.

    `values` must be a list without NaN, `count` its length.
    """
    if count <= 1:
        return float('nan')

    variance_sum = sum(((x - mean) ** 2) for x in values)
    variance = variance_sum / (count - 1)
    return variance ** 0.5
