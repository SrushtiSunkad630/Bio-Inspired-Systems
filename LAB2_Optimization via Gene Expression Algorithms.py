import random
import numpy as np

# ----------------------
# 1. Define the Problem
# Sphere function (minimize sum of squares)
def objective_function(x):
    return sum(xi ** 2 for xi in x)

# ----------------------
# 2. Initialize Parameters
POP_SIZE = 10          # smaller population size
NUM_GENES = 3          # dimensionality of solution
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 5        # fewer generations for small output
SEARCH_SPACE = (-5, 5) # gene values range

# ----------------------
# 3. Initialize Population
def init_population():
    return [np.random.uniform(SEARCH_SPACE[0], SEARCH_SPACE[1], NUM_GENES).tolist() for _ in range(POP_SIZE)]

# ----------------------
# 4. Evaluate Fitness (lower is better)
def fitness(individual):
    return objective_function(individual)

# ----------------------
# 5. Selection (Tournament Selection)
def select(population):
    k = 3
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: fitness(ind))
    return selected[0]

# ----------------------
# 6. Crossover (Single Point)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_GENES - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]

# ----------------------
# 7. Mutation
def mutate(individual):
    for i in range(NUM_GENES):
        if random.random() < MUTATION_RATE:
            individual[i] = random.uniform(SEARCH_SPACE[0], SEARCH_SPACE[1])
    return individual

# ----------------------
# 8. Gene Expression (identity mapping here)
def express(individual):
    return individual  # In real GEA, encoding/decoding would occur

# ----------------------
# 9. Iterate
population = init_population()
best_solution = min(population, key=fitness)

for gen in range(GENERATIONS):
    new_population = []
    while len(new_population) < POP_SIZE:
        parent1 = select(population)
        parent2 = select(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(express(child1))
        child2 = mutate(express(child2))
        new_population.extend([child1, child2])
    
    population = sorted(new_population, key=fitness)[:POP_SIZE]
    current_best = population[0]
    if fitness(current_best) < fitness(best_solution):
        best_solution = current_best

    print(f"Generation {gen+1}: Best Fitness = {fitness(best_solution):.6f}")

# ----------------------
# 10. Output Best Solution
print("\nBest Solution Found:", best_solution)
print("Best Fitness:", fitness(best_solution))



OUTPUT
Generation 1: Best Fitness = 0.882722
Generation 2: Best Fitness = 0.882722
Generation 3: Best Fitness = 0.440984
Generation 4: Best Fitness = 0.440984
Generation 5: Best Fitness = 0.440984

Best Solution Found: [-0.21860594328748206, 0.18519227131556715, -0.5990816220714512]
Best Fitness: 0.4409835256993896
