import numpy as np
import matplotlib.pyplot as plt


ray = np.array([1, 0, 0])

rotation_angle = np.pi / 4
rotation_vector = np.array([1, 2, 1])

q_s = np.cos(rotation_angle / 2)
q_a = np.sin(rotation_angle / 2) * rotation_vector

rotated_ray = q_s * q_s * ray + 2 * (q_s * np.cross(q_a, ray) + q_a * np.dot(q_a, ray)) - np.dot(q_a, q_a) * ray

print(f"Original ray: {ray}")
print(f"Rotated ray: {rotated_ray}")

x = np.zeros([1, 2])
y = np.zeros([1, 2])
z = np.zeros([1, 2])

u = np.array([ray[0], rotated_ray[0]])
v = np.array([ray[1], rotated_ray[1]])
w = np.array([ray[2], rotated_ray[2]])


ax = plt.figure().add_subplot(projection='3d')


ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
ax.quiver(0, 0, 0, rotation_vector[0], rotation_vector[1], rotation_vector[2], length=0.1, normalize=True)

plt.show()