import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Specify the path to your CSV file
filepath1 = '/home/student/g.csv'
filepath2 = '/home/student/x.csv'

# Read the CSV file into a DataFrame
df1 = pd.read_csv(filepath1)
df2 = pd.read_csv(filepath2)

G = (df1.to_numpy()[:, 1:4])
x_p = (df2.to_numpy()[:, 1:4])


#print(x_p)

# #Cost function
# def cost_function(x_e, x_p, G):

#     diff = (x_e - x_p)/np.linalg.norm(x_e - x_p) - G/np.linalg.norm(G)
#     #diff = (x_e - x_p) - G
#     return np.sum(np.linalg.norm(diff, axis=1)**2)

def cost_function(x_e, x_p, G):
    gazevec=x_e-x_p
    dotscore=np.sum(np.multiply(G, gazevec))
    denom=np.linalg.norm(G)*np.linalg.norm(gazevec)
    diff = (dotscore/denom) - 1
    return np.sum(np.linalg.norm(diff)**2)

# Initial guess for x_e
initial_guess = np.array([0, 0, 0])

# Find x_e using SciPy's minimize function
result = minimize(cost_function, initial_guess, args=(x_p, G))
x_e_optimal = result.x

print(x_e_optimal)
