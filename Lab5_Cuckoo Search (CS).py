import numpy as np
import math

# Problem data (example)
weights = np.array([10, 20, 30, 40, 15])
values = np.array([60, 100, 120, 240, 70])
W = 50  # knapsack capacity
num_items = len(weights)

def fitness(solution):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)
    if total_weight > W:
        return 0  # penalize infeasible solutions
    else:
        return total_value

def levy_flight(Lambda=1.5, dim=num_items):
    sigma1 = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
              (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=dim)
    v = np.random.normal(0, sigma2, size=dim)
    step = u / abs(v) ** (1 / Lambda)
    return step

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cuckoo_search_knapsack(n=20, Pa=0.25, max_iter=500):
    # Initialize nests randomly with binary vectors
    nests = np.random.randint(2, size=(n, num_items))
    fitness_vals = np.array([fitness(nest) for nest in nests])

    best_idx = np.argmax(fitness_vals)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness_vals[best_idx]

    for t in range(max_iter):
        for i in range(n):
            # Generate new solution by Levy flight
            step = levy_flight()
            new_solution_continuous = nests[i] + step
            probs = sigmoid(new_solution_continuous)
            new_solution = np.where(np.random.rand(num_items) < probs, 1, 0)

            new_fitness = fitness(new_solution)

            if new_fitness > fitness_vals[i]:
                nests[i] = new_solution
                fitness_vals[i] = new_fitness

                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_nest = new_solution.copy()

        # Abandon fraction Pa of worst nests and replace with new random solutions
        abandon_indices = np.where(np.random.rand(n) < Pa)[0]
        for idx in abandon_indices:
            nests[idx] = np.random.randint(2, size=num_items)
            fitness_vals[idx] = fitness(nests[idx])

            if fitness_vals[idx] > best_fitness:
                best_fitness = fitness_vals[idx]
                best_nest = nests[idx].copy()

    return best_nest, best_fitness

# Run the algorithm
best_solution, best_value = cuckoo_search_knapsack()
print("Best solution (item selection vector):", best_solution)
print("Total value:", best_value)
print("Total weight:", np.sum(best_solution * weights))


############OUTPUT################
Best solution (item selection vector): [1 0 0 1 0]
Total value: 300
Total weight: 50
