from collections import deque

class Vertex:
    def __init__(self, value):
        self.value = value      # имя / значение вершины
        self.neighbors = []     # соседние вершины

    def add_neighbor(self, vertex):
        self.neighbors.append(vertex)

    def __repr__(self):
        return f"Vertex({self.value})"
vertex_a = Vertex("A")
vertex_b = Vertex("B")
vertex_c = Vertex("C")
vertex_d = Vertex("D")
vertex_e = Vertex("E")

vertex_a.add_neighbor(vertex_b)
vertex_a.add_neighbor(vertex_c)
vertex_b.add_neighbor(vertex_d)
vertex_c.add_neighbor(vertex_d)
vertex_d.add_neighbor(vertex_e)


def bfs(start):
    visited = set()
    queue = deque([start])  # кладём сразу вершину, а не список

    while queue:
        vertex = queue.popleft()

        if vertex in visited:
            continue

        visited.add(vertex)
        print(vertex)  # или собираем куда-то, или просто посещаем

        for neighbor in vertex.neighbors:
            if neighbor not in visited:
                queue.append(neighbor)

    return visited

def dfs(vertex, visited=None):
    if visited is None:
        visited = set()

    if vertex in visited:
        return

    print(vertex.value)
    visited.add(vertex)

    for neighbor in vertex.neighbors:
        dfs(neighbor, visited)

graph = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C", "E"],
    "E": ["D"]
}

def find_path(graph, start, end):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == end:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                new_path = path + [neighbor]
                queue.append(new_path)

    return None
print(find_path(graph, "A", "E"))
print(bfs(vertex_a))