def fib(n):
    """
    Return the first Fibonacci number above n.
    Iteratively calculate Fibonacci numbers until it finds one
    greater than n, which it then returns.
    Parameters
    ----------
    n : integer
      The minimum threshold for the desired Fibonacci number.
    Returns
    -------
    b : integer
      The first Fibonacci number greater than the input, `n`.
    Examples
    --------
    >>> fib.fib(1)
    2
    >>> fib.fib(3)
    5
    """
    a = 0
    b = 1
    while b <= n:
        a, b = b, a + b
    return b
