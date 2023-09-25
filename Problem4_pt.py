import numpy as np
import math

def quarter_circle_points(R, x1, y1, x2, y2):
    """
    Given R, and the coordinates of two points, return the coordinates of the points on a quarter circle
    with center at (R, y1) and radius R, from (x1, y1) to (x2, y2).
    """
    # Initial and final points
    point1 = [x1, y1]
    point2 = [x2, y2]

    # Calculate the angle spanned by the segment from point1 to point2
    delta_y = y2 - y1
    delta_x = x2 - x1
    start_angle = math.degrees(math.atan2(y1 - R, x1 - R))
    end_angle = math.degrees(math.atan2(y2 - R, x2 - R))

    # Calculate the coordinates of the 4 intermediate points based on equal divisions of the segment angle
    points = [point1]
    for i in range(1, 5):
        theta = math.radians(start_angle + (end_angle - start_angle) * i / 5)
        x = R + R * math.cos(theta)
        y = R + R * math.sin(theta)
        points.append([x,y])
    points.append(point2)

    return points


a = 60
b = 60
c = 40
d = 60
R = 6
import matplotlib.pyplot as plt
# cal theta

sin_t = 1/R*(b/2 -c/2 -R)
theda = 0
cos_t = np.sqrt(1 - (-R + b/2 - c/2)**2/R**2)
# Compute the circle's center
center_x = R 
center_y = R + c/2
# Compute points on the circle
theta = np.linspace(np.pi+np.pi/2,  np.pi+np.pi/2-(np.pi/2+theda), 10)
theta = np.linspace(np.pi+np.pi/2-(np.pi/2+theda), np.pi+np.pi/2, 10)

x_circle = center_x + R * np.cos(theta)
y_circle = center_y + R * np.sin(theta)

# Plot the given points and the circle
plt.figure(figsize=(10, 6))
plt.plot(x_circle, y_circle, color='red', label='Circle with center $(R \cos(t), R + \frac{c}{2})$')

############################################


outputlist = []

outputlist.append([-a, -b/2])
outputlist.append([-a, b/2])
outputlist.append([0, b/2])

for x,y in zip(x_circle,y_circle):
    outputlist.append([x,y])
#####################################################

# Compute the circle's center
center_x = R
center_y = -R - c/2
# Compute points on the circle
theta = np.linspace(np.pi/2,  np.pi/2+(np.pi/2+theda), 10)
#theta = np.linspace(np.pi/2+(np.pi/2+theda),  np.pi/2, 10)

x_circle = center_x + R * np.cos(theta)
y_circle = center_y + R * np.sin(theta)

# Plot the given points and the circle

plt.plot(x_circle, y_circle, color='red', label='Circle with center $(R \cos(t), R + \frac{c}{2})$')
##############################################################

outputlist.append([d, c/2])
outputlist.append([d, -c/2])

for x,y in zip(x_circle,y_circle):
    outputlist.append([x,y])

outputlist.append([0, -b/2])




for i in outputlist:
    print(i)

# Extract x and y coordinates
x, y = zip(*outputlist)


plt.scatter(x, y, color='blue', marker='o')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Plot of Given Points')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()