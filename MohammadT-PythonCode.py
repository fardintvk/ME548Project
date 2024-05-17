import numpy as np
from scipy.signal import ss2tf

# System 1
A1 = np.array([[-0.05, -400, 0],
               [0, -0.5, 0.0001],
               [0, 0, -0.05]])
B1 = np.array([[1],
               [0],
               [0]])
C1 = np.array([[1, 0, 0]])

# Convert to transfer function
G1_num, G1_den = ss2tf(A1, B1, C1, np.zeros((1, 1)))

# System 2
A2 = np.array([[-0.05, -400, 0],
               [0, -0.5, 0.0001],
               [0, 0, -0.05]])
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








import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

### Response without any input
# Define the system matrices
A = np.array([[-0.05, -400, 0],
              [0, -0.5, 0.0001],
              [0, 0, -0.05]])
B = np.array([[1, 0],
              [0, 0],
              [0, 1]])
C = np.array([[1, 0, 0]])

# Define the system function dx/dt = Ax + Bu
def system(t, x):
    return np.dot(A, x)

# Time vector
t = np.linspace(0, 50, 1000)

# Initial condition
X0 = [400, 0, 0]

# Solve the differential equation using solve_ivp
sol = solve_ivp(system, [t[0], t[-1]], X0, t_eval=t)

# Extracting the output for plotting
output = sol.y[0]  # Assuming you want the first output for plotting

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(sol.t, output)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Response of the Dynamic System without inputs and control system')
plt.grid(True)
plt.show()

### Response by using ug input
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system matrices
A = np.array([[-0.05, -400, 0],
              [0, -0.5, 0.0001],
              [0, 0, -0.05]])
B = np.array([[1, 0],
              [0, 0],
              [0, 1]])
C = np.array([[1, 0, 0]])

# Define the system function dx/dt = Ax + Bu
def system(t, x, u):
    return np.dot(A, x) + np.dot(B, u)

# Time vector
t = np.linspace(0, 50, 1000)

# Define the input signal U = [1; 0]
def input_signal(t):
    return np.array([10, 0])

# Solve the differential equation with input using solve_ivp
sol = solve_ivp(lambda t, x: system(t, x, input_signal(t)), [t[0], t[-1]], [400, 0, 0], t_eval=t)

# Extracting the output for plotting
output = sol.y[0]  # Assuming you want the first output for plotting

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(sol.t, output)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Response of the Dynamic System with Input U=[10;0] (Patient use glucose from foods but he has not use any madication.)')
plt.grid(True)
plt.show()


### Response by using both ug and ui inputs
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system matrices
A = np.array([[-0.05, -400, 0],
              [0, -0.5, 0.0001],
              [0, 0, -0.05]])
B = np.array([[1, 0],
              [0, 0],
              [0, 1]])
C = np.array([[1, 0, 0]])

# Define the system function dx/dt = Ax + Bu
def system(t, x, u):
    return np.dot(A, x) + np.dot(B, u)

# Time vector
t = np.linspace(0, 50, 1000)

# Define the input signal U = [1; 0]
def input_signal(t):
    return np.array([10, 1])

# Solve the differential equation with input using solve_ivp
sol = solve_ivp(lambda t, x: system(t, x, input_signal(t)), [t[0], t[-1]], [400, 0, 0], t_eval=t)

# Extracting the output for plotting
output = sol.y[0]  # Assuming you want the first output for plotting

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(sol.t, output)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Response of the Dynamic System with Input U=[10;1] (Patient use both glucose foods and medication)')
plt.grid(True)
plt.show()

import numpy as np
from scipy.signal import cont2discrete

# Define the continuous-time system matrices
A = np.array([[-0.05, -400, 0],
              [0, -0.5, 0.0001],
              [0, 0, -0.05]])
B = np.array([[1, 0],
              [0, 0],
              [0, 1]])
C = np.array([[1, 0, 0]])

# Sampling time (Ts) for discretization
Ts = 0.1  # Adjust as needed

# Discretize the system using cont2discrete
Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, 0), Ts, method='zoh')

# Print the discretized system matrices
print("Discretized System Matrices:")
print("Ad =", Ad)
print("Bd =", Bd)
print("Cd =", Cd)


import cvxpy as cp
import numpy as np

# Define system matrices
A = np.array([[9.95e-1, -3.89e1, -1.96e-4],
              [0, 9.51e-1, 9.73e-6],
              [0, 0, 9.95e-1]])

B = np.array([[9.98e-2, -6.57e-6],
              [0, 4.91e-7],
              [0, 9.98e-2]])

C = np.array([[1, 0, 0]])

# Initial condition
X0 = np.array([[400], [0], [0]])

# Define the length of the prediction horizon
N = 10  # You can adjust this based on your application

# Define the weighting factor sigma
sigma = 1  # You can adjust this based on the importance of minimizing medication input

# Define optimization variables
u = cp.Variable((B.shape[1], N))  # Control inputs over the prediction horizon

# Define the state and output variables over the prediction horizon
x = cp.Variable((A.shape[0], N+1))  # States
y = cp.Variable((C.shape[0], N+1))  # Outputs

# Define the objective function (cost function)
cost = 0
for i in range(N):
    cost += sigma * (cp.sum_squares(u[:, i]) + cp.sum_squares(x[:, i] - 87))

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


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define system matrices
A = np.array([[9.95e-1, -3.89e1, -1.96e-4],
              [0, 9.51e-1, 9.73e-6],
              [0, 0, 9.95e-1]])

B = np.array([[9.98e-2, -6.57e-6],
              [0, 4.91e-7],
              [0, 9.98e-2]])

C = np.array([[1, 0, 0]])

# Initial condition
X0 = np.array([[400], [0], [0]])

# Define the length of the prediction horizon
N = 50  # You can adjust this based on your application

# Define the weighting factor sigma
sigma = 1  # You can adjust this based on the importance of minimizing medication input

# Define optimization variables
u = cp.Variable((B.shape[1], N))  # Control inputs over the prediction horizon

# Define the state and output variables over the prediction horizon
x = cp.Variable((A.shape[0], N+1))  # States
y = cp.Variable((C.shape[0], N+1))  # Outputs

# Define the objective function (cost function)
cost = 0
for i in range(N):
    cost += sigma * (cp.sum_squares(u[:, i]) + cp.sum_squares(x[:, i] - 87))

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
plt.xlabel('Time Steps')
plt.ylabel('System Output')
plt.title('System Response with MPC Control')
plt.legend()
plt.grid(True)
plt.show()
