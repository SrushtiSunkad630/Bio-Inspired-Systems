import numpy as np
import matplotlib.pyplot as plt

class GreyWolfOptimizer:
    def __init__(self, obj_func, lb, ub, dim, num_wolves=20, max_iter=100):
        self.obj_func = obj_func      # Objective function
        self.lb = np.array(lb)        # Lower bound
        self.ub = np.array(ub)        # Upper bound
        self.dim = dim                # Dimension of problem
        self.num_wolves = num_wolves
        self.max_iter = max_iter

        # Initialize wolves randomly within search space
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
       
        # Initialize alpha, beta, delta wolves
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
       
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
       
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')
       
        self.convergence_curve = []
   
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.num_wolves):
                # Ensure wolves stay inside bounds
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
               
                fitness = self.obj_func(self.positions[i])
               
                # Update alpha, beta, delta
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
           
            # Coefficient a decreases linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)
           
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # Update position of wolf i in dimension j
                    self.positions[i, j] = (X1 + X2 + X3) / 3

            self.convergence_curve.append(self.alpha_score)
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.alpha_score}")
       
        return self.alpha_pos, self.alpha_score

# Example usage
if __name__ == "__main__":
    # Example objective function: Sphere function
    def sphere(x):
        return np.sum(x**2)

    dim = 5
    lb = [-10] * dim
    ub = [10] * dim

    gwo = GreyWolfOptimizer(obj_func=sphere, lb=lb, ub=ub, dim=dim, num_wolves=30, max_iter=5)
    best_pos, best_score = gwo.optimize()

    print("Best position found:", best_pos)
    print("Best objective value:", best_score)

    plt.plot(gwo.convergence_curve)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("GWO Convergence Curve")
    plt.show()


############OUTPUT###############
Iteration 1/50, Best Fitness: 42.79424996701728
Iteration 2/50, Best Fitness: 12.500799853658568
Iteration 3/50, Best Fitness: 8.298738015697063
Iteration 4/50, Best Fitness: 3.920207133633615
Iteration 5/50, Best Fitness: 1.2226806180901382
Iteration 6/50, Best Fitness: 0.33591137932150095
Iteration 7/50, Best Fitness: 0.14635012301763084
Iteration 8/50, Best Fitness: 0.044272842167895544
Iteration 9/50, Best Fitness: 0.030495292717057178
Iteration 10/50, Best Fitness: 0.013167212117360516
Iteration 11/50, Best Fitness: 0.00462294065750525
Iteration 12/50, Best Fitness: 0.004443247891892386
Iteration 13/50, Best Fitness: 0.0023615849510666333
Iteration 14/50, Best Fitness: 0.0012681548062805233
Iteration 15/50, Best Fitness: 0.000335204706973059
Iteration 16/50, Best Fitness: 0.000335204706973059
Iteration 17/50, Best Fitness: 0.00015665927860789518
Iteration 18/50, Best Fitness: 6.124724493681782e-05
Iteration 19/50, Best Fitness: 2.8687297235493365e-05
Iteration 20/50, Best Fitness: 2.290278640075217e-05
Iteration 21/50, Best Fitness: 1.1038502498064416e-05
Iteration 22/50, Best Fitness: 6.970144816855817e-06
Iteration 23/50, Best Fitness: 5.62588421165653e-06
Iteration 24/50, Best Fitness: 3.085234496825378e-06
Iteration 25/50, Best Fitness: 2.178618121233763e-06
Iteration 26/50, Best Fitness: 1.062751109431824e-06
Iteration 27/50, Best Fitness: 6.798589594733175e-07
Iteration 28/50, Best Fitness: 5.116334092283641e-07
Iteration 29/50, Best Fitness: 3.877340072775356e-07
Iteration 30/50, Best Fitness: 2.534734171887928e-07
Iteration 31/50, Best Fitness: 1.8956016285602229e-07
Iteration 32/50, Best Fitness: 1.5870632202955642e-07
Iteration 33/50, Best Fitness: 1.0143451258928606e-07
Iteration 34/50, Best Fitness: 7.050206297731235e-08
Iteration 35/50, Best Fitness: 5.6192189661778585e-08
Iteration 36/50, Best Fitness: 4.5096529009903717e-08
Iteration 37/50, Best Fitness: 3.8111726771464614e-08
Iteration 38/50, Best Fitness: 3.094101713957885e-08
Iteration 39/50, Best Fitness: 2.8665902809959634e-08
Iteration 40/50, Best Fitness: 2.7157287028991157e-08
Iteration 41/50, Best Fitness: 2.3565149852575313e-08
Iteration 42/50, Best Fitness: 2.0705964538010005e-08
Iteration 43/50, Best Fitness: 1.942724433437514e-08
Iteration 44/50, Best Fitness: 1.667300472162348e-08
Iteration 45/50, Best Fitness: 1.6112683199170945e-08
Iteration 46/50, Best Fitness: 1.515325016469644e-08
Iteration 47/50, Best Fitness: 1.4414032997815537e-08
Iteration 48/50, Best Fitness: 1.379047054274129e-08
Iteration 49/50, Best Fitness: 1.341236500112782e-08
Iteration 50/50, Best Fitness: 1.3266480235990753e-08
Best position found: [ 5.44510240e-05 -5.36893850e-05  4.23171830e-05 -5.06662989e-05
 -5.53280971e-05]
Best objective value: 1.3266480235990753e-08
