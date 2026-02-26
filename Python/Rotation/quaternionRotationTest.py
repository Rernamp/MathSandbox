import numpy as np


ray = np.array([1, 0, 0])

rotation_angle = np.pi / 2
rotation_vector = np.array([1, 0, 0])

q_s = np.cos(rotation_angle / 2)
q_a = np.sin(rotation_angle / 2) * rotation_vector

rotated_ray = q_s * q_s * ray + 2 * (q_s * np.cross(q_a, ray) + q_a * np.dot(q_a, ray)) - np.dot(q_a, q_a) * ray

print(f"Original ray: {ray}")
print(f"Rotated ray: {rotated_ray}")