import numpy as np
from scipy.signal import ss2tf, cont2discrete
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import cvxpy as cp

plt.rcParams['font.family'] = 'Times New Roman'

# Define Ts for discretization
Ts = 1  # You can adjust this as needed

# Define the length of the prediction horizon
N = 70  # You can adjust this based on your application

# System 1
A1 = np.array([[-0.05, -400, 0],
               [0, -0.5, 0.0001],
               [0, 0, 0]])
B1 = np.array([[1],
               [0],
               [0]])
C1 = np.array([[1, 0, 0]])

# Convert to transfer function
G1_num, G1_den = ss2tf(A1, B1, C1, np.zeros((1, 1)))

# System 2
A2 = np.array([[-0.05, -400, 0],
               [0, -0.5, 0.0001],
               [0, 0, 0]])
B2 = np.array([[0],
               [0],
               [1]])
C2 = np.array([[1, 0, 0]])

# Convert to transfer function
G2_num, G2_den = ss2tf(A2, B2, C2, np.zeros((1, 1)))

# Print transfer function matrices
print("Transfer Function Matrix G1(S):")
print("Numerator:", G1_num)
print("Denominator:", G1_den)
print("\nTransfer Function Matrix G2(S):")
print("Numerator:", G2_num)
print("Denominator:", G2_den)


### Response without any input

# Define the system matrices
A = np.array([[-0.05, -400, 0],
              [0, -0.5, 0.0001],
              [0, 0, 0]])
B = np.array([[1, 0],
              [0, 0],
              [0, 1]])
C = np.array([[1, 0, 0]])

# Define the system function dx/dt = Ax + Bu
def system(t, x):
    return np.dot(A, x)

# Time vector
t = np.linspace(0, 150, 1000)

# Initial condition
X0 = np.array([[400], [0], [0]])

# Solve the differential equation using solve_ivp
sol = solve_ivp(system, [t[0], t[-1]], X0.flatten(), t_eval=t)

# Extracting the output for plotting
output = sol.y[0]  # Assuming you want the first output for plotting

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(sol.t, output)
plt.axhline(y=87, color='r', linestyle='--', label='Desired Value')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Response of the Dynamic System without inputs and control system', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

### Response by using ug input

# Define the system function dx/dt = Ax + Bu
def system(t, x, u):
    return np.dot(A, x) + np.dot(B, u)

# Time vector
t = np.linspace(0, 150, 1000)

# Define the input signal U = [30; 0]
def input_signal(t):
    return np.array([30, 0])

# Solve the differential equation with input using solve_ivp
sol = solve_ivp(lambda t, x: system(t, x, input_signal(t)), [t[0], t[-1]], X0.flatten(), t_eval=t)

# Extracting the output for plotting
output = sol.y[0]  # Assuming you want the first output for plotting

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(sol.t, output)
plt.axhline(y=87, color='r', linestyle='--', label='Desired Value')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Response of the Dynamic System with Input U=[30;0]', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()


# Define the input signal U = [30; 5]
def input_signal(t):
    return np.array([30, 3])


# Define the time grid
t = [0, 150]  # Adjust as needed

# Define the time grid with finer resolution
t_eval = np.linspace(t[0], t[-1], 1000)

# Solve the differential equation with input using solve_ivp with the finer time grid
sol = solve_ivp(lambda t, x: system(t, x, input_signal(t)), [t[0], t[-1]], X0.flatten(), t_eval=t_eval)

# Extracting the output for plotting
output = sol.y[0]

# Modify the output to ensure it's non-negative
output_modified = np.where(output < 0, 0, output)

# Plot the modified response
plt.figure(figsize=(10, 6))
plt.plot(sol.t, output_modified)
plt.axhline(y=87, color='r', linestyle='--', label='Desired Value')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Smooth Response of the Dynamic System with Input U=[30;5]', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()



# Discretize the system using cont2discrete
Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, 0), Ts, method='zoh')

# Print the discretized system matrices
print("Discretized System Matrices:")
print("Ad =", Ad)
print("Bd =", Bd)
print("Cd =", Cd)


# Define discrete system matrices
A = np.array([[9.95e-1, -3.89e1, -1.96e-4],
              [0, 9.51e-1, 9.73e-6],
              [0, 0, 1]])

B = np.array([[9.98e-2, -6.57e-6],
              [0, 4.91e-7],
              [0, 1]])

C = np.array([[1, 0, 0]])

# Define the weighting factor sigma
sigma = 0.8  # You can adjust this based on the importance of minimizing medication input

# Define optimization variables
u = cp.Variable((B.shape[1], N))  # Control inputs over the prediction horizon

# Define the state and output variables over the prediction horizon
x = cp.Variable((A.shape[0], N+1))  # States
y = cp.Variable((C.shape[0], N+1))  # Outputs

# Define the desired value for the output
desired_value = 87

# Modify the objective function (cost function) to penalize deviation from the desired value
cost = 0
for i in range(N):
    cost += sigma * (cp.sum_squares(u[:, i]) + cp.sum_squares(x[:, i] - desired_value) + (y[:, i] - desired_value)**2)

# Define system dynamics constraints
constraints = []
for i in range(N):
    constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]
    constraints += [y[:, i] == C @ x[:, i]]

# Define initial condition constraint
constraints += [x[:, 0] == X0.flatten()]

# Formulate the optimization problem
prob = cp.Problem(cp.Minimize(cost), constraints)

# Solve the optimization problem
prob.solve()

# Get the optimal control inputs
optimal_u = u.value

# Print the optimal control inputs
print("Optimal control inputs:")
print(optimal_u)


# Define optimization variables
u = cp.Variable((B.shape[1], N))  # Control inputs over the prediction horizon

# Define the state and output variables over the prediction horizon
x = cp.Variable((A.shape[0], N+1))  # States
y = cp.Variable((C.shape[0], N+1))  # Outputs

# Define the objective function (cost function)
cost = 0
for i in range(N):
    cost += sigma * (cp.sum_squares(u[:, i]) + cp.sum_squares(x[:, i] - desired_value) + (y[:, i] - desired_value)**2)

# Define system dynamics constraints
constraints = []
for i in range(N):
    constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]
    constraints += [y[:, i] == C @ x[:, i]]

# Define initial condition constraint
constraints += [x[:, 0] == X0.flatten()]

# Formulate the optimization problem
prob = cp.Problem(cp.Minimize(cost), constraints)

# Solve the optimization problem
prob.solve()

# Get the optimal control inputs
optimal_u = u.value

# Calculate the optimal value of the cost function
optimal_cost = cost.value

# Print the optimal value of the cost function
print("Optimal value of cost function:", optimal_cost)

# Plot the system response
time_steps = np.arange(N+1)
plt.figure(figsize=(10, 6))
plt.plot(time_steps, x[0, :].value, label='System Output')
plt.axhline(y=desired_value, color='r', linestyle='--', label='Desired Value')
plt.xlabel('Time Steps')
plt.ylabel('System Output')
plt.title('System Response with MPC Control', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()
