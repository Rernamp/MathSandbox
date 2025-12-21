import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def error(x, target):
    f = x - target
    # df = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # return f, df
    return f

targetPosition = np.array([1, 1, 1])
print(f"Target position {targetPosition}")

initialPosition = np.array([1, 2, 3])
print(f"Initial position {targetPosition}")

# sol = root(error, initialPosition, args=targetPosition, jac=True, method="lm")
sol = root(error, initialPosition, args=targetPosition)

print(f"Solution: x {sol.x}, value : {sol.fun}")