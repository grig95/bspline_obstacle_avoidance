import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def bsplines_casadi(knot, deg):
    """
    Python/CasADi version of the MATLAB function [b,knot] = bsplines_casadi(knot, deg).
    
    knot : list or array of knot points
    deg  : spline degree
    
    Returns
    -------
    b    : a list-of-lists of CasADi Function objects, representing B-spline basis functions. note that for the n-th order function the index is n-1 and b indexed first by order, then by k/l
    knot : the sorted knot array
    """

    # 1) Sort and ensure real values of the knot vector
    knot = sorted(float(x) for x in knot)
    m = len(knot)

    # 2) Check for the minimum knot size based on spline degree
    if m < deg + 2:
        raise ValueError(
            "The knot size (m) and the degree (deg) must satisfy m >= deg + 2."
        )

    # 3) Define a local Heaviside function via CasADi if_else
    #    heaviside(z) = 1 if z >= 0, else 0
    def heaviside(z):
        return ca.if_else(z >= 0, 1, 0)

    # 4) Define the CasADi symbolic variable
    x = ca.SX.sym("x", 1)

    # 5) Prepare a nested list b[k][i], where
    #    k goes from 0 to deg (instead of 1 to deg+1 as in MATLAB),
    #    i will range accordingly for each k.
    b = []
    # For each k in [0..deg], we will have up to (m - k) functions
    for k_ in range(deg + 1):
        b.append([None] * (m - k_))

    # 6) Build the B-spline functions of order 1 (equivalent to k=1 in MATLAB)
    #    b[0][i] corresponds to b{k=1}{i+1} in MATLAB
    #    for i in [0..m-2] <-> MATLAB i in [1..m-1]
    for i in range(m - 1):
        # heaviside(x - knot[i]) - heaviside(x - knot[i+1])
        expr = heaviside(x - knot[i]) - heaviside(x - knot[i + 1])
        b[0][i] = ca.Function(
            f"f_1_{i+1}",  # Function name: "f_1_i" in MATLAB was i in [1..m-1]
            [x], 
            [expr],
        )

    # 7) Apply the recursive B-spline relation for orders 2..(deg+1)
    #    (eq. (2.1) from the original code reference)
    #    In MATLAB: for k=2:deg+1, for i=1:m-k
    #    Here:      for k_ in [1..deg], i in [0..(m - 1 - k_)]
    for k_ in range(1, deg + 1):
        # k_ in [1..deg], which corresponds to B-spline order k_+1 in MATLAB terms
        # so the function name is "f_{k_+1}_{i+1}" to match original naming
        for i in range(m - 1 - k_):
            # We need to evaluate b[k_-1][i] and b[k_-1][i+1] at the symbolic x.
            # In MATLAB: b{k-1}{i}(x) => b[k_-2][i-1] in zero-based
            # But we also need to *call* that CasADi function to get a symbolic expr.

            # First sub-term: 
            denom1 = knot[i + k_] - knot[i]
            if denom1 != 0:
                left_val = b[k_ - 1][i](x)[0] * (x - knot[i]) / denom1
            else:
                left_val = 0

            # Second sub-term:
            denom2 = knot[i + k_ + 1] - knot[i + 1]
            if denom2 != 0:
                right_val = b[k_ - 1][i + 1](x)[0] * (knot[i + k_ + 1] - x) / denom2
            else:
                right_val = 0

            expr = left_val + right_val

            b[k_][i] = ca.Function(
                f"f_{k_+1}_{i+1}",
                [x],
                [expr],
            )

    return b, knot

def bspline_derivative(b, knots, l, d):
    # Define symbolic variable
    t = ca.SX.sym("t")

    # First term
    if knots[l + d] - knots[l] == 0.0:
        term1 = 0
    else:
        b1 = b[d - 1][l](t)[0]  # Evaluate function at t
        term1 = b1 / (knots[l + d] - knots[l])

    # Second term
    if knots[l + d + 1] - knots[l + 1] == 0.0:
        term2 = 0
    else:
        b2 = b[d - 1][l + 1](t)[0]  # Evaluate function at t
        term2 = b2 / (knots[l + d + 1] - knots[l + 1])

    deriv_expr = d * (term1 - term2)

    # Return as a CasADi Function
    return ca.Function(f"db_{l}_{d}", [t], [deriv_expr])

def approx_integral(b, n, d, sample_count=100):
    samples = np.linspace(0.0, 1.0, sample_count)
    P = ca.SX.sym("P", 2, n)
    b_deriv = [bspline_derivative(b, knots, i, d) for i in range(n)]
    sum = 0.0
    for sample in samples:
        bderiv_eval = ca.vertcat(*[b_deriv[i](sample) for i in range(n)])
        curve_dot = P @ bderiv_eval  # shape (2, 1)
        sum += ca.dot(curve_dot, curve_dot)
    cost = sum / sample_count
    cost_func = ca.Function("sampled_bspline_cost", [P], [cost])
    return cost_func



def solve(d, knots, obstacles, start, finish):
    n = len(knots)-d-1
    solver = ca.Opti()
    P = solver.variable(2, n)
    epsilon = solver.variable(1)
    separator = solver.variable(1, 2)

    #prepare function
    b, knots = bsplines_casadi(knots, d)
    print(f'len(b[d]): {len(b[d])}')

    #integration stuff
    ''' am incercat si cu ca.integrator dar degeaba :'(
    B_dot = [bspline_derivative(b, knots, i, d) for i in range(n)]
    B_dot_eval = ca.vertcat(*[B_dot[i](t) for i in range(n)]) # shape (n, 1)
    curve_derivative = P @ B_dot_eval # shape (2, 1)
    squared_norm = ca.dot(curve_derivative, curve_derivative)
    x = ca.SX.sym("x")
    ode = {'x': x, 'p': ca.vec(P), 'ode': squared_norm}
    integrator = ca.integrator("bspline_cost_integrator", "cvodes", ode, 0.0, 1.0, {})
    P_sym = ca.SX.sym("P_input", 2, n)
    sol = integrator(x0=0, p=ca.vec(P_sym))
    cost_expr = sol["xf"]
    integration_cost_func = ca.Function("bspline_cost", [P_sym], [cost_expr])
    '''


    #add goal
    solver.minimize(approx_integral(b, n, d)(P) + 100*epsilon)
    
    # add constranints
    for obst in obstacles:
        for point in obst:
            solver.subject_to(separator @ point <= 1.0)
    for i in range(n):
        solver.subject_to(separator @ P[:, i] >= 1.0)
        solver.subject_to(P[:, i] <=epsilon)
        solver.subject_to(P[:, i] >= -epsilon)
    def z(t):
        return ca.sum2(P @ ca.vertcat(*[b[d][i](t) for i in range(n)]))
    solver.subject_to(z(0.0) == start)
    print([b[d][i](1.0) for i in range(n)])
    solver.subject_to(ca.norm_2(z(0.99) - finish) <= 0.5)
    
    # attach a solver
    solver.solver('ipopt')
    # and solve the optimization problem
    sol = solver.solve()
    return P, sol


def plot_bspline_and_obstacles(bspline_basis_functions, P_sol, obstacles, d, start, finish):
    """
    bspline_basis_functions: list of lists of CasADi basis function objects, indexed by [d-1][i]
    P_sol: (2, n+1) numpy array of control points
    obstacles: list of convex polygons, each represented by list of (x, y) tuples
    d: the order of the B-spline (degree + 1)
    """
    basis = bspline_basis_functions[d - 1]
    n = P_sol.shape[1]

    # Evaluate the spline over a fine grid
    t_vals = np.linspace(0, 1, 300)
    curve_points = []

    for t in t_vals:
        point = np.zeros(2)
        for i in range(n):
            b = basis[i](t)
            point += np.array(P_sol[:, i]) * float(b)
        curve_points.append(point)

    curve_points = np.array(curve_points)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
    ax.plot(P_sol[0, :], P_sol[1, :], 'ro--', label='Control Polygon')

    # Plot each obstacle
    for obs in obstacles:
        obs = np.array(obs)
        obs_closed = np.vstack([obs, obs[0]])  # Close the polygon
        ax.plot(obs_closed[:, 0], obs_closed[:, 1], 'k-', label='Obstacle')
        ax.fill(obs[:, 0], obs[:, 1], color='gray', alpha=0.3)
    
    # Plot start and finish
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(finish[0], finish[1], 'mo', markersize=10, label='Finish')

    ax.set_aspect('equal')
    ax.legend()
    plt.title("B-spline Curve and Obstacles")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()



m = 14
d = 3
n = m - d - 1
knots = np.concatenate((
    np.zeros(d),
    np.linspace(0, 1, m - 2*d),
    np.ones(d))
)
print(knots)
obstacles = [[np.array([0.2, 0.2]), np.array([0.4, 0.4]), np.array([0.4, 0.2])], 
             [np.array([1.8, 1.2]), np.array([2.4, 2.5]), np.array([2.5, 2.1])]]
start = np.array([0.1, 0.1])
finish = np.array([2.2, 1.3])
P, sol = solve(d, knots, obstacles, start, finish)
P_sol = sol.value(P)
plot_bspline_and_obstacles(bsplines_casadi(knots, d)[0], P_sol, obstacles, d, start, finish)