import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt
from matplotlib import cm
from argparse import ArgumentParser


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
MATRIX[0, 0] = 0
MATRIX[0, 2**N] =0
MATRIX[2**N, 2**N] = 0
MATRIX[2**N, 0] = 0


def in_matrix(size, point):
    return 0 <= point[0] < size and 0 <= point[1] < size


def point_value(points, step):
    v = sum(points) / len(points) + (2**step) * SIGMA * normal()
    return v


def square_step(size, point, delta, step, matrix):
    points = [
        (point[0], point[1] - delta),
        (point[0] + delta, point[1]),
        (point[0], point[1] + delta),
        (point[0] - delta, point[1]),
    ]
    correct_points = [matrix[p[0], p[1]] for p in points if in_matrix(size, p)]
    matrix[point[0], point[1]] = sum(correct_points) / len(correct_points) + (2**(step-1)) * SIGMA * normal()
    return matrix


def diamond_step(size, point, delta, step, matrix):
    points = [
        (point[0] - delta, point[1] - delta),
        (point[0] + delta, point[1] - delta),
        (point[0] + delta, point[1] + delta),
        (point[0] - delta, point[1] + delta)
    ]
    correct_points = [matrix[p[0], p[1]] for p in points if in_matrix(size, p)]
    matrix[point[0], point[1]] = sum(correct_points) / len(correct_points) + (2**step) * SIGMA * normal()
    return matrix


def poz(n, matrix):
    size = 2**n+1
    points = [(size//2, size//2)]
    delta = size // 2
    for step in range(n, 0, -1):
        # print(step)
        next_points = []
        for point in points:
            diamond_step(size, point, delta, step, matrix)
            next_points.extend([
                (point[0], point[1] - delta),
                (point[0] + delta, point[1]),
                (point[0], point[1] + delta),
                (point[0] - delta, point[1]),
            ])
        points = list(set([point for point in next_points if in_matrix(size, point)]))

        next_points = []
        for point in points:
            square_step(size, point, delta, step - 1, matrix)
            next_points.extend([
                (point[0] - delta//2, point[1] - delta//2),
                (point[0] + delta//2, point[1] - delta//2),
                (point[0] + delta//2, point[1] + delta//2),
                (point[0] - delta//2, point[1] + delta//2)
            ])
        delta //= 2
        points = list(set([point for point in next_points if in_matrix(size, point)]))
        step -= 1
    return matrix


fig1 = plt.figure(num=1)
ax = plt.axes(projection='3d')

X = np.arange(0, 2**N+1, 1)
Y = np.arange(0, 2**N+1, 1)
X, Y = np.meshgrid(X, Y)
Z = poz(N, matrix=MATRIX)

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
