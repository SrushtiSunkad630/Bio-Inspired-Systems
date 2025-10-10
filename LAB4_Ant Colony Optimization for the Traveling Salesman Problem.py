import random
import math

# --- Helper Functions ---
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def route_length(route):
    return sum(distance(route[i], route[(i + 1) % len(route)]) for i in range(len(route)))

# --- Main ACO Function ---
def ant_colony_tsp(cities, ants=5, alpha=1, beta=2, rho=0.5, iterations=20):
    n = len(cities)
    pheromone = [[1 for _ in range(n)] for _ in range(n)]
    dist = [[distance(cities[i], cities[j]) for j in range(n)] for i in range(n)]

    best_route, best_distance = None, float('inf')

    for _ in range(iterations):
        all_routes = []

        for _ in range(ants):
            route = [random.randint(0, n-1)]
            while len(route) < n:
                i = route[-1]
                probs = []
                for j in range(n):
                    if j not in route:
                        tau = pheromone[i][j] ** alpha
                        eta = (1 / dist[i][j]) ** beta
                        probs.append((j, tau * eta))
                next_city = random.choices([p[0] for p in probs], weights=[p[1] for p in probs])[0]
                route.append(next_city)

            all_routes.append(route)

            L = route_length([cities[i] for i in route])
            if L < best_distance:
                best_distance, best_route = L, route

        # Evaporate pheromone
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - rho)

        # Add pheromone for best routes
        for route in all_routes:
            L = route_length([cities[i] for i in route])
            for i in range(n):
                a, b = route[i], route[(i + 1) % n]
                pheromone[a][b] += 1 / L

    return best_route, best_distance


cities = [(0,0), (1,5), (5,1), (3,4), (6,3)]
best_route, best_dist = ant_colony_tsp(cities)
print("Best Route:", best_route)
print("Best Distance:", best_dist)



OUTPUT
Best Route: [3, 1, 0, 2, 4]
Best Distance: 17.832452642353527
