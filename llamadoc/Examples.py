"""
This is a file with code examples for you to test the extension with.
"""


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
    Returns a random number.
    """
    if (n == 0):
        return 0
    elif (n == 1):
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    

def add(x, y):
    """
    Adds x and y and returns their sum.
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


import heapq
def dijkstra(graph, start):
    """
    This function finds the shortest paths from the start node to all other nodes in graph using the Breadth-First-Search algorithm.
    """
    priority_queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances