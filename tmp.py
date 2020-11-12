import queue

class Edge:
    def __init__(self, to, cost):
        self.to = to
        self.cost = cost


class Node:
    def __init__(self, team_number):
        self.team_number = team_number
        self.edges = []

    def add_edge(self, to, cost):
        self.edges.append(Edge(to, cost))


def dijkstra(nodes, start, end):
    distance = [1e31 for _ in range(len(nodes))]
    node_teams = team_numbers.copy()
    inq = [False for _ in range(len(nodes))]
    last = [0 for _ in range(len(nodes))]
    num = [0 for _ in range(len(nodes))]

    num[start] = 1
    distance[start] = 0
    inq[start] = True

    qe = queue.PriorityQueue()
    qe.put([0, start])

    while not qe.empty():
        top_distance, top_index = qe.get()
        inq[top_index] = False
        top_node = nodes[top_index]
        for edge in top_node.edges:
            to = edge.to
            cost = edge.cost

            if distance[to] > cost + top_distance:
                last[to] = top_index
                num[to] = num[top_index]
                distance[to] = cost + top_distance
                node_teams[to] = node_teams[top_index] + nodes[to].team_number
                if not inq[to]:
                    inq[to] = True
                    qe.put([distance[to], to])


            elif distance[to] == cost + top_distance:
                num[to] += num[top_index]
                if node_teams[to] < node_teams[top_index] + nodes[to].team_number:
                    last[to] = top_index
                    node_teams[to] = node_teams[top_index] + nodes[to].team_number
                    if not inq[to]:
                        inq[to] = True
                        qe.put([distance[to], to])

    print(num[end], node_teams[end])
    # print(last)
    path = []
    s = end
    while s != start:
        path.append(s)
        s = last[s]
    path.reverse()
    print(start, end="")
    if start != end:
        print(" ", end="")
    for s in path[:-1]:
        print(s, end=" ")
    print(path[-1])


N, M, S, D = [int(i) for i in input().split()]
team_numbers = [int(i) for i in input().split()]

nodes = [Node(i) for i in team_numbers]

for _ in range(M):
    a, b, c = [int(i) for i in input().split()]
    nodes[a].add_edge(b, c)
    nodes[b].add_edge(a, c)

dijkstra(nodes, S, D)

