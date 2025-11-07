import random

# Parameters
POP_SIZE = 6
GENES = 5        # 5 bits to represent 0-31
GENERATIONS = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1

# Fitness function: f(x) = x^2
def fitness(binary_str):
    x = int(binary_str, 2)
    return x * x

# Create initial population of random 5-bit binary strings
def create_population():
    population = []
    for _ in range(POP_SIZE):
        individual = ''.join(random.choice('01') for _ in range(GENES))
        population.append(individual)
    return population

# Selection: Tournament Selection of size 2
def tournament_selection(pop):
    i1, i2 = random.sample(pop, 2)
    return i1 if fitness(i1) > fitness(i2) else i2

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENES - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

# Mutation: Bit flip mutation
def mutate(individual):
    new_ind = ''
    for bit in individual:
        if random.random() < MUTATION_RATE:
            new_ind += '1' if bit == '0' else '0'
        else:
            new_ind += bit
    return new_ind

# Main GA function
def genetic_algorithm():
    population = create_population()
    best_individual = None
    best_fitness = -1

    for gen in range(1, GENERATIONS + 1):
        new_population = []

        # Evaluate and keep track of best
        for ind in population:
            ind_fit = fitness(ind)
            if ind_fit > best_fitness:
                best_fitness = ind_fit
                best_individual = ind

        # Print best in current generation
        print(f"Generation {gen}: Best Individual = {best_individual} (x={int(best_individual, 2)}), Fitness = {best_fitness}")

        # Create new generation
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:POP_SIZE]

    print(f"\nBest solution found: {best_individual} (x={int(best_individual, 2)}), Fitness = {best_fitness}")

if __name__ == "__main__":
    genetic_algorithm()


OUTPUT
Generation 1: Best Individual = 11101 (x=29), Fitness = 841
Generation 2: Best Individual = 11101 (x=29), Fitness = 841
Generation 3: Best Individual = 11101 (x=29), Fitness = 841
Generation 4: Best Individual = 11101 (x=29), Fitness = 841
Generation 5: Best Individual = 11101 (x=29), Fitness = 841

Best solution found: 11101 (x=29), Fitness = 841
