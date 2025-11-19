import heapq
import itertools
from geneticalgorithm import geneticalgorithm as ga
import numpy as np

""" Runs a Genetic Algorithm to optimize traffic flow in a network graph. """

def dijkstra(graph, source, destination):
    distances = {node: float("inf") for node in graph}
    distances[source] = 0
    priority_queue = [(0, source)]  # (distance, node)
    previous_nodes = {node: None for node in graph}
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node == destination:
            break
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    return distances[destination], path


def loss_function(X):
    graph = {
        "A": {"B": X[0], "C": X[1]},
        "B": {"A": X[2], "C": X[3], "D": X[4]},
        "C": {"A": X[5], "B": X[6], "D": X[7]},
        "D": {"B": X[8], "C": X[9]},
    }

    safety_limits = {i: {j: 30 for j in graph[i]} for i in graph}
    safety_limits["A"]["B"] = 50
    safety_limits["A"]["C"] = 50

    source = "A"
    destination = "D"

    people_using_path = {i: {j: 0 for j in graph[i]} for i in graph}

    for _ in range(100):
        distance, path = dijkstra(graph, source, destination)
        for a, b in itertools.pairwise(path):
            graph[a][b] += 1
            people_using_path[a][b] += 1

    loss = 0
    for start in people_using_path:
        for end in people_using_path[start]:
            loss += (
                max(people_using_path[start][end] - safety_limits[start][end], 0) ** 2
            )
    return loss


varbound = np.array([[0, 200]] * 10)

model = ga(
    function=loss_function,
    dimension=10,
    variable_type="int",
    variable_boundaries=varbound,
)
model.run()
print(model.param)
