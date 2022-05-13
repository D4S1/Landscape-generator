import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt
from matplotlib import cm
from argparse import ArgumentParser
from random import randint


parser = ArgumentParser()
# implementacja flag
parser.add_argument("-N", "--size", help="rozmiar macierzy")
parser.add_argument("-s", "--sigma", help="stopien gorzystosci")
parser.add_argument("-mpsf", "--map_save_file", help="nazwa pliku do zapisu mapy")
parser.add_argument("-ssf", "--surface_save_file", help="nazwa pliku do zapisu powierzchni")
parser.add_argument("-c", "--map_color", help="kolor wykresow")
parser.add_argument("-mxsf", "--matrix_save_file", help="nazwa pliku do zapisu macierzy")
parser.add_argument("-mxlf", "--matrix_load_file", help="nazwa pliku do wczytania macierzy")
parser.add_argument("-bgs", "--begin_step", help="krok poczatkowy uzupelniania macierzy")
parser.add_argument("-ends", "--end_step", help="krok koncowy uzupelniania macierzy")

args = parser.parse_args()


# wczytanie danych, jeżeli jakieś są
N = int(args.size) if args.size else 10
SIGMA = float(args.sigma) if args.sigma else 1
map_save_file = args.map_save_file if args.map_save_file else ""
surface_save_file = args.surface_save_file if args.surface_save_file else ""
matrix_save_file = args.matrix_save_file if args.matrix_save_file else ""
map_color = args.map_color if args.map_color else 'terrain'
matrix_load_file = args.matrix_load_file if args.matrix_load_file else ''
start_step = int(args.begin_step) if args.begin_step else None
end_step = int(args.end_step) if args.end_step else 1

# print jakby ktoś chciał znać ustawione parametry
# print(f"""
# N -> {N}
# SIGMA -> {SIGMA}
# map file -> {map_save_file}
# surface file -> {surface_save_file}
# matrix file -> {matrix_save_file}
# map color -> {map_color}
# matrix load file -> {matrix_load_file}
# start step-> {start_step}
# end step -> {end_step}
# """)

# sprawdzenie czy krok początkowy i końcowy (jeżeli któryś z tych parametrów został podany) są z dobrego przedziału
if start_step and (N < start_step or start_step < 1):
    raise Exception("krok początkowy z poza zakresu")

if N < end_step or end_step < 1:
    raise Exception("krok początkowy z poza zakresu")

# sprawdzenie czy początkowy krok jest podawany jednocześnie z plikiem do wczytania macierzy
# nie można podać tylko jednego z tych dwóch parametrów
if start_step and not matrix_load_file or matrix_load_file and not start_step:
    raise Exception("podaj krok początkowy oraz plik do wczytania macierzy")

# jak podajemy end_step to program nie wyswietla nam wykresów tylko zapisuje częsciowo wypełnioną macierz
# do podanego pliku, więc trzeba sprawdzić czy jeżeli podano end_step to czy podano także plik do zapisu macierzy
if end_step != 1 and not matrix_save_file:
    raise Exception("podaj krok końcowy oraz plik do zapisania macierzy")


# WCZYTYWANIE ALBO GENEROWANIE MACIERZY
if matrix_load_file:
    MATRIX = np.loadtxt(matrix_load_file, dtype='i', delimiter=' ')
else:
    MATRIX = np.ones((2**N+1, 2**N+1))
    # losowanie rogów
    MATRIX[0, 0] = randint(-100, 100)
    MATRIX[0, 2**N] = randint(-100, 100)
    MATRIX[2**N, 2**N] = randint(-100, 100)
    MATRIX[2**N, 0] = randint(-100, 100)


def in_matrix(size, point):
    """
    Zwraca czy dany punkt zanjduje się w kwadratowej macierzy o rozmiarze size
    :param size: int
    :param point: tuple (x, y_
    :return: Boolean
    """
    return 0 <= point[0] < size and 0 <= point[1] < size


def convert_point(size, point):
    """
    zwraca niezmieniony punkt, jeżeli znajduje się on w macierzy. W przeciwnym razie zwraca zawinięty punkt
    czyli np jak y < 0 czyli wyjdziemy nad górną krawędź macierzy to funkcja zwróci punkt o y odpowiednio oddalonym
    od dolnej krawędzi y = size - y
    :param size: int
    :param point: tuple (x, y)
    :return: tuple (x, y)
    """
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
    """
    zwraca wartość -> średnią z punktow + zaburzenie
    :param points: list (wartości punktów sąsiednich)
    :param step: int (obecny krok)
    :return: int
    """
    return sum(points) / len(points) + (2**step) * SIGMA * normal()


def square_step(size, point, delta, step, matrix):
    """
    zmienia wartość w punkcie point (mechanizm square step)
    :param size: int
    :param point: tuple
    :param delta: int
    :param step: int
    :param matrix: matrix
    :return: matrix
    """
    points = [
        (point[0], point[1] - delta),
        (point[0] + delta, point[1]),
        (point[0], point[1] + delta),
        (point[0] - delta, point[1]),
    ]
    correct_points = [convert_point(size, p) for p in points]
    values = [matrix[x, y] for x, y in correct_points]
    matrix[point[0], point[1]] = point_value(values, step)
    return matrix


def diamond_step(size, point, delta, step, matrix):
    """
        zmienia wartość w punkcie point (mechanizm diamond step)
        :param size: int
        :param point: tuple
        :param delta: int
        :param step: int
        :param matrix: matrix
        :return: matrix
        """
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


def poz(n, matrix, start_step, end_step):
    """
    uzupełnia w pętli macierz matrix
    :param n: int
    :param matrix: matrix
    :param start_step: int
    :param end_step: int
    :return: matrix
    """
    size = 2**n+1
    begin = start_step if start_step else n

    # tworzymy listę punktów do diamond step'u
    for step in range(begin, end_step-1, -1):
        delta = size // (2**(n+1-step))
        points = []
        for x in range(delta, size-delta, 2*delta):
            for y in range(delta, size-delta, 2*delta):
                points.append((x, y))

        next_points = []
        for point in points:
            # ustawiamy wartość dla punktu(diamond)
            diamond_step(size, point, delta, step, matrix)
            # dodajemy punkty, dla których będziemy wykonywać square step
            next_points.extend([
                (point[0], point[1] - delta),
                (point[0] + delta, point[1]),
                (point[0], point[1] + delta),
                (point[0] - delta, point[1]),
            ])
        # usuwamy powtórzenia
        points = list(set(next_points))
        for point in points:
            # ustawiamy wartość dla punktu(square)
            square_step(size, point, delta, step - 1, matrix)
    return matrix


# jeżeli został podany end_step zapisujemy macierz do pliku
if end_step != 1:
    matrix = poz(N, matrix=MATRIX, start_step=start_step, end_step=end_step)
    np.savetxt(matrix_save_file, matrix, fmt='%.5f')

else:
    X = np.arange(0, 2 ** N + 1, 1)
    Y = np.arange(0, 2 ** N + 1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = poz(N, matrix=MATRIX, start_step=start_step, end_step=end_step)

    # okno z powierzchnią
    fig1 = plt.figure(num=1)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap(map_color), linewidth=0, antialiased=False)
    fig1.colorbar(surf, shrink=0.5, aspect=5)

    # jeżeli podany plik do zapisu powierzchni zapisujemy ją jako plik png, jak nie otwieramy okno
    if surface_save_file:
        plt.savefig(fname=surface_save_file, format='png')
        plt.close()

    fig1 = None
    # Okno z mapą
    fig2 = plt.figure(num=2)
    plt.imshow(Z, cmap=cm.get_cmap(map_color), extent=[0, 2**N+1, 0, 2**N+1])

    # jeżeli podany plik do zapisu mapy zapisujemy ją jako plik png, jak nie otwieramy okno
    if map_save_file:
        plt.savefig(fname=map_save_file, format='png')
        plt.close()

    plt.show()

    # jeżeli podany plik do zapisu macirzy zapisujemy ją jako plik png, jak nie to nic nie robimy
    if matrix_save_file:
        np.savetxt(matrix_save_file, Z, fmt='%.5f')
