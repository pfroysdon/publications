import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import pulp

def solve_tsp(num_cities=10, seed=1):
    np.random.seed(seed)
    # Generate random city coordinates in a 100x100 region
    city_coords = np.random.rand(num_cities, 2) * 100
    
    # Plot cities
    plt.figure()
    plt.scatter(city_coords[:, 0], city_coords[:, 1], s=100, c='k')
    plt.title("City Locations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    for i in range(num_cities):
        plt.text(city_coords[i, 0] + 1, city_coords[i, 1], str(i+1), fontsize=12, color='blue')
    plt.grid(True)
    plt.show()
    
    # Distance matrix (Euclidean distances)
    D = squareform(pdist(city_coords))
    np.fill_diagonal(D, 0)
    
    n = num_cities
    # Create MILP problem
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)
    
    # Decision variables: x[i][j] = 1 if edge from i to j is used
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat='Binary') for j in range(n)] for i in range(n)]
    # Continuous variables for MTZ constraints: u[i] for i=1,...,n-1 (we fix u[0]=0)
    u = [None] * n
    u[0] = 0  # fixed
    for i in range(1, n):
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n-1, cat='Continuous')
    
    # Objective: minimize total tour length
    prob += pulp.lpSum(D[i, j] * x[i][j] for i in range(n) for j in range(n))
    
    # Constraints:
    # 1. Each city has exactly one outgoing edge.
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n)) == 1
    
    # 2. Each city has exactly one incoming edge.
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(n)) == 1
    
    # 3. No self-loops.
    for i in range(n):
        prob += x[i][i] == 0
    
    # 4. MTZ subtour elimination constraints for i,j = 1,...,n-1, i != j.
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1
    
    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise ValueError("The solver did not find an optimal solution.")
    
    # Extract solution
    X_sol = np.array([[pulp.value(x[i][j]) for j in range(n)] for i in range(n)])
    X_sol = np.rint(X_sol)  # Round to 0 or 1
    
    print(f"Optimal tour length: {pulp.value(prob.objective):.2f}")
    print("Decision variable matrix X (rows: from city, columns: to city):")
    print(X_sol)
    
    # Extract tour starting from city 0
    tour = [0]
    current_city = 0
    for _ in range(n-1):
        next_city = np.where(X_sol[current_city, :] == 1)[0][0]
        tour.append(next_city)
        current_city = next_city
    tour.append(tour[0])
    
    print("Optimal tour: " + " -> ".join(str(city+1) for city in tour))
    
    # Visualize tour
    plt.figure()
    plt.plot(city_coords[:, 0], city_coords[:, 1], 'ko', markersize=10)
    for i in range(n):
        plt.text(city_coords[i, 0]+1, city_coords[i, 1], str(i+1), fontsize=15, color='blue')
    for i in range(n):
        frm = tour[i]
        to = tour[i+1]
        plt.plot([city_coords[frm, 0], city_coords[to, 0]], [city_coords[frm, 1], city_coords[to, 1]], 'r-', linewidth=2)
    plt.title("Optimal TSP Tour")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    solve_tsp(num_cities=10, seed=1)
