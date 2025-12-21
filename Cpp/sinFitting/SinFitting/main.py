import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data
x_data = np.linspace(0, 2 * np.pi, 100)
y_data = 721 * np.sin(1.5 * x_data + 0.5) + 6.0 + 100 * np.random.normal(size=100)

# Define the sine function
def sine_func(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Initial guess
initial_guess = [(y_data.max() - y_data.min()) / 2, 10.0, 0.0, 0.0]

# Fit the sine function to the data
params, params_covariance = curve_fit(sine_func, x_data, y_data, p0=initial_guess)

# Extract the fitted parameters
A_fit, B_fit, C_fit, D_fit = params

# Print the fitted parameters
print(f"Fitted Parameters: A={A_fit}, B={B_fit}, C={C_fit}, D={D_fit}")

y_fit = sine_func(x_data, A_fit, B_fit, C_fit, D_fit)
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, label='Fitted Sine Function', color='red')
plt.legend()
plt.show()