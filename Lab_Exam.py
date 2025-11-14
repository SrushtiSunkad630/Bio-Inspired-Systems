import numpy as np
import random

# ------------------------
# PARAMETERS
# ------------------------
NUM_NODES = 20
NUM_ANTS = 10
ALPHA = 1.0            # pheromone importance
BETA = 3.0             # distance importance
EVAPORATION = 0.4
Q = 100

random.seed(42)
np.random.seed(42)

# ------------------------
# NODE DEPLOYMENT
# ------------------------
nodes = np.random.randint(0, 100, (NUM_NODES, 2))  # random (x,y)
BASE_STATION = np.array([50, 50])

# ------------------------
# DISTANCE MATRIX
# ------------------------
def euclidean(a, b):
    return np.linalg.norm(a - b)

dist = np.zeros((NUM_NODES, NUM_NODES))
for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        dist[i][j] = euclidean(nodes[i], nodes[j])

# Distance from node → base station
dist_to_bs = np.array([euclidean(nodes[i], BASE_STATION) for i in range(NUM_NODES)])

# ------------------------
# INITIAL PHEROMONES
# ------------------------
pheromone = np.ones((NUM_NODES, NUM_NODES))

# ------------------------
# ACO CONSTRUCTION OF PATH
# ------------------------
def choose_next(current, visited):
    candidates = [j for j in range(NUM_NODES) if j not in visited]
    if not candidates:
        # no candidate left (shouldn't normally happen) -> return random node
        return random.randint(0, NUM_NODES - 1)
    scores = []
    for j in candidates:
        tau = pheromone[current][j] ** ALPHA
        eta = (1.0 / (dist[current][j] + 1e-9)) ** BETA
        scores.append(tau * eta)
    scores = np.array(scores)
    if scores.sum() == 0:
        # fallback: uniform over candidates
        return random.choice(candidates)
    probs = scores / scores.sum()
    chosen = np.random.choice(len(candidates), p=probs)
    return int(candidates[chosen])

# ------------------------
# COST OF FULL PATH
# ------------------------
def path_cost(path):
    cost = 0.0
    for i in range(len(path) - 1):
        cost += dist[path[i]][path[i + 1]]
    cost += dist_to_bs[path[-1]]
    return cost

# ------------------------
# MAIN ACO LOOP
# ------------------------
ITERATIONS = 5
MAX_HOPS = 5  # path length

for iteration in range(ITERATIONS):
    all_paths = []
    all_costs = []

    for ant in range(NUM_ANTS):
        start = random.randint(0, NUM_NODES - 1)
        visited = [start]

        # Build full multi-hop route up to MAX_HOPS
        while len(visited) < MAX_HOPS:
            next_node = choose_next(visited[-1], visited)
            visited.append(next_node)

        c = path_cost(visited)
        all_paths.append(visited)
        all_costs.append(c)

    # Get best of this iteration
    best_index = int(np.argmin(all_costs))
    best_path = [int(x) for x in all_paths[best_index]]
    best_cost = all_costs[best_index]

    # Evaporate
    pheromone *= (1 - EVAPORATION)

    # Deposit pheromones along the best path
    if best_cost > 0:
        deposit = Q / best_cost
        for i in range(len(best_path) - 1):
            a = best_path[i]
            b = best_path[i + 1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

    # Print output in the requested format
    print(f"Iteration {iteration}:")
    print("Best Path:", best_path)
    print("Cost:", round(best_cost, 2))
    print()


OUTPUT
Iteration 0:
Best Path: [4, 17, 0, 16, 12]
Cost: 90.29

Iteration 1:
Best Path: [0, 16, 12, 1, 18]
Cost: 94.49

Iteration 2:
Best Path: [6, 11, 2, 13, 19]
Cost: 81.43

Iteration 3:
Best Path: [6, 11, 2, 13, 19]
Cost: 81.43

Iteration 4:
Best Path: [3, 4, 17, 0, 16]
Cost: 97.52
