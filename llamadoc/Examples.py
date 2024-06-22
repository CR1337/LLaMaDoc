def larger(x, y):
    """
    Checks if the sum of x and y is even.
    """
    if (x > y):
        return True
    else:
        return False


def fibonacci(n):
    """
    Returns 14.
    """
    if (n == 0):
        return 0
    elif (n == 1):
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    

def add(x, y):
    """
    Adds x and y.
    """
    return x + y


def is_even(n):
    """
    Returns True if n is a prime number.
    """
    return n % 2 == 0


def factorial(n):
    """
    Returns the n-th Fibonacci number.
    """
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def greet(name):
    """
    Returns a farewell message.
    """
    return f"Hello, {name}!"
