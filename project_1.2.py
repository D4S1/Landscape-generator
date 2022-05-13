import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt
from matplotlib import cm
from argparse import ArgumentParser
from random import randint


parser = ArgumentParser()
# FLAGS IMPLEMENTATION
parser.add_argument("-N", "--size", help="rozmiar macierzy")
parser.add_argument("-s", "--sigma", help="stopien gorzystosci")
parser.add_argument("-mpf", "--mapfile", help="nazwa pliku do zapisu mapy")
parser.add_argument("-sf", "--surfacefile", help="nazwa pliku do zapisu powierzchni")
parser.add_argument("-c", "--mapcolor", help="kolor map")
parser.add_argument("-mxf", "--matrixfile", help="nazwa pliku do zapisu macierzy")

args = parser.parse_args()


N = int(args.size) if args.size else 7
SIGMA = float(args.sigma) if args.sigma else 0.66
map_file = args.mapfile if args.mapfile else ""
surface_file = args.surfacefile if args.surfacefile else ""
matrix_file = args.matrixfile if args.matrixfile else ""
map_color = args.mapcolor if args.mapcolor else 'terrain'
print(f"""
N -> {N}
SIGMA -> {SIGMA}
map file -> {map_file}
surface file -> {surface_file}
matrix file -> {matrix_file}
map color -> {map_color}
""")

MATRIX = np.ones((2**N+1, 2**N+1))
for x, y in [(0, 0), (0, 2**N), (2**N, 2**N), (2**N, 0)]:
    MATRIX[x, y] = randint(-100, 100)


def in_matrix(size, point):
    return 0 <= point[0] < size and 0 <= point[1] < size


def convert_point(size, point):
    x, y = point[0], point[1]
    # idea
    # return point if in matrix else moved point eg. size = 5 (-2, 2) -> (2, 2)
    # mechanism 2**N - x or y it depends which one is out of range
    if x < 0:
        x += size - 1
    elif size <= x:
        x -= size
    elif y < 0:
        y += size - 1
    elif size <= y:
        y -= size
    return x, y


def point_value(points, step):
    return sum(points) / len(points) + (2**step) * SIGMA * normal(0, 1)


def square_step(size, point, delta, step, matrix):
    points = [
        (point[0], point[1] - delta),
        (point[0] + delta, point[1]),
        (point[0], point[1] + delta),
        (point[0] - delta, point[1]),
    ]
    correct_points = [convert_point(size, p) for p in points]
    values = [matrix[x, y] for x, y in correct_points]
    matrix[point[0], point[1]] = point_value(values, step - 1)
    return matrix


def diamond_step(size, point, delta, step, matrix):
    points = [
        (point[0] - delta, point[1] - delta),
        (point[0] + delta, point[1] - delta),
        (point[0] + delta, point[1] + delta),
        (point[0] - delta, point[1] + delta)
    ]
    correct_points = [convert_point(size, p) for p in points]
    values = [matrix[x, y] for x, y in correct_points]
    matrix[point[0], point[1]] = point_value(values, step)
    return matrix


def poz(n, matrix):
    size = 2**n+1
    points = [(size//2, size//2)]
    for step in range(n, 0, -1):
        delta = size // (2 ** (n + 1 - step))

        next_points = []
        for point in points:

            diamond_step(size, point, delta, step, matrix)
            next_points.extend([
                (point[0] - delta, point[1]),
                (point[0] + delta, point[1]),
                (point[0], point[1] + delta),
                (point[0], point[1] - delta)
                          ])
        points = list(set([point for point in next_points if in_matrix(size, point)]))

        next_points = []
        for point in points:
            square_step(size, point, delta, step, matrix)
            next_points.extend([
                (point[0] + delta//2, point[1] + delta//2),
                (point[0] - delta//2, point[1] - delta//2),
                (point[0] + delta//2, point[1] - delta//2),
                (point[0] - delta//2, point[1] + delta//2),
            ])

        points = list(set([point for point in next_points if in_matrix(size, point)]))
    return matrix


X = np.arange(0, 2**N+1, 1)
Y = np.arange(0, 2**N+1, 1)
X, Y = np.meshgrid(X, Y)
Z = poz(N, matrix=MATRIX)


fig1 = plt.figure(num=1)
ax = plt.axes(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap(map_color), linewidth=0, antialiased=False)
fig1.colorbar(surf, shrink=0.5, aspect=5)

if surface_file:
    plt.savefig(fname=surface_file, format='png')
    plt.close()

fig2 = plt.figure(num=2)
plt.imshow(Z, cmap=cm.get_cmap(map_color), extent=[0, 2**N+1, 0, 2**N+1])
if map_file:
    plt.savefig(fname=map_file, format='png')
    plt.close()

plt.show()

if matrix_file:
    np.savetxt(matrix_file, Z, fmt='%.5f')
