import heapq

class PuzzleNode:
    def __init__(self, state, parent=None, move=None, depth=0, cost=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(str(self.state))

class NPuzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.size = len(goal_state)
        self.goal_positions = self.GetGoalPos()

    def GetGoalPos(self):
        goal_positions = {}
        for i, row in enumerate(self.goal_state):
            for j, value in enumerate(row):
                goal_positions[value] = (i, j)
        return goal_positions

    def heuristic(self, node):
        distance = 0
        for i, row in enumerate(node.state):
            for j, value in enumerate(row):
                if value != 0:
                    goal_i, goal_j = self.goal_positions[value]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def get_neighbors(self, node):
        neighbors = []
        for i, row in enumerate(node.state):
            for j, value in enumerate(row):
                if value == 0:
                    empty_i, empty_j = i, j
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  
        for move_i, move_j in moves:
            new_i, new_j = empty_i + move_i, empty_j + move_j
            if 0 <= new_i < self.size and 0 <= new_j < self.size:
                new_state = [row[:] for row in node.state]
                new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
                neighbors.append(PuzzleNode(new_state, parent=node, move=(move_i, move_j), depth=node.depth+1, cost=node.depth+1+self.heuristic(node)))
        return neighbors

    def reconstruct_path(self, node):
        path = []
        while node.parent:
            path.append(node.move)
            node = node.parent
        return list(reversed(path))

    def a_star_algorithm(self):
        OpenList = []
        ClosedList = set()
        initial_node = PuzzleNode(self.initial_state)
        heapq.heappush(OpenList, initial_node)

        while OpenList:
            current_node = heapq.heappop(OpenList)

            if current_node.state == self.goal_state:
                return self.reconstruct_path(current_node)

            ClosedList.add(current_node)
            neighbors = self.get_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor in ClosedList:
                    continue
                if neighbor not in OpenList:
                    heapq.heappush(OpenList, neighbor)
                else:
                    existing_neighbor = OpenList[OpenList.index(neighbor)]
                    if neighbor.cost < existing_neighbor.cost:
                        existing_neighbor.cost = neighbor.cost
                        existing_neighbor.parent = neighbor.parent
                        existing_neighbor.move = neighbor.move

        return None

initial_state = [[2, 8, 1],[0, 4, 3],[7, 6, 5]]

goal_state = [[1, 2, 3],[8, 0, 4],[7, 6, 5]]

solve = NPuzzle(initial_state, goal_state)
sol = solve.a_star_algorithm()
print("Solution:", sol)
