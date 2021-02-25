import math
import numpy as np
import random


def read_data_from_file(file):
    try:
        f = open(file, "r")
        n = int(f.readline())
        epsilon = pow(10, -int(f.readline()))
        a = []
        for i in range(n):
            line = [float(x) for x in f.readline().split(" ")]
            a.append(line)
        b = [float(x) for x in f.readline().split(" ")]
        if not np.all(np.linalg.eigvals(a) > 0):
            exit("Matricea nu este pozitiv definita!")
        return n, epsilon, a, b
    except OSError:
        print("Error while reading data!")


def read_data_from_console():
    n = int(input("Introduceti numarul de linii a matricei patratice:"))
    epsilon = pow(10, -int(input("Introduceti m pentru valoarea lui epsilon de forma 10^(-m):")))
    # Initialize matrix
    a = []
    print("Introduceti randurile matricei:")
    for i in range(n):
        a1 = []
        for j in input().split(" "):
            x = float(j)
            a1.append(x)
        a.append(a1)
    print("Introduceti vectorul termenilor liberi:")
    b = [float(x) for x in input().split(" ")]
    if not np.all(np.linalg.eigvals(a) > 0):
        exit("Matricea nu este pozitiv definita!")
    return n, epsilon, a, b


def generate_random_matrix():
    n = random.randint(10, 100)
    epsilon = pow(10, -random.randint(10, 20))
    a = [[0 for _ in range(n)] for _ in range(n)]
    s = [0 for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if i != j:
                a[i][j] = a[j][i] = random.uniform(-10, 10)
    for i in range(n):
        for j in range(n):
            s[i] += abs(a[i][j])
        a[i][i] = s[i] + random.uniform(epsilon, 10);
    b = [random.uniform(1, 1000) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            print(a[i][j], end=" ")
        print()
    return n, epsilon, a, b


def cholesky_decomposition(a, n, epsilon):
    for i in range(n):
        for j in range(0, i + 1):
            if i == j:
                s = 0
                for k in range(i):
                    s += a[i][k] ** 2
                print(a[i][j], s)
                if a[i][j] - s > 0:
                    a[i][j] = math.sqrt(a[i][j] - s)
                else:
                    print("nu se poate face radical")
                    exit(0)
            else:
                s = 0
                for k in range(j):
                    s += a[i][k] * a[j][k]
                if abs(a[j][j]) > epsilon:
                    a[i][j] = (a[i][j] - s) / a[j][j]
                else:
                    print("nu se poate face impartirea")
                    exit(0)
    print("Descompunere Cholesky: ", a)


def determinant(a):
    """
    :param a: matrice inferior triunghiulara din descompunerea lui A, notata si L
    :return: determinantul matricei A prin formula det(A)=det(L)*det(LT)
    """
    det = 1
    for i in range(n):
        det *= a[i][i]
    return det ** 2


def forward_substitution(L, b, epsilon):
    """
    Forward substiution method solves the system Ly=b, where L is a lower triangular matrix
    :param L: matrice inferior triunghiulara din descompunerea lui A
    :param b: factorul drept al sistemului Ax=b/Ly=b
    """
    y = [0 for _ in range(len(b))]
    y[0] = b[0] / L[0][0]
    for i in range(1, len(y)):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        if abs(L[i][i]) > epsilon:
            y[i] = (b[i] - s) / L[i][i]
        else:
            print("nu se poate face impartirea")
            exit(0)
    return y


def backward_substitution(L, y, epsilon):
    """
    Backward substiution method solves the system LTx=y, where LT is a upper triangular matrix (L transpose)
    :param L: matrice inferior triunghiulara din descompunerea lui A
    :param y: valorea calculata la forward_substitution rezolvand sistemului Ly=b
    """
    n = len(y)
    x = [0 for _ in range(n)]
    x[n - 1] = y[n - 1] / L[n - 1][n - 1]
    for i in reversed(range(n - 1)):
        s = 0
        for j in range(i + 1, n):
            s += L[j][i] * x[j]
        if abs(L[i][i]) > epsilon:
            x[i] = (y[i] - s) / L[i][i]
        else:
            print("nu se poate face impartirea")
            exit(0)
    return x


def solve_eq(a, b, epsilon):
    y = forward_substitution(a, b, epsilon)
    print("Solutia ecuatiei Ly=b:", y)
    x = backward_substitution(a, y, epsilon)
    return x


def norm(a, x, d, b):
    """
    :param a: matricea initiala simetrica si pozitiv definita, modificata ulterior in descompunerea Cholesky
    :param x: solutia sistemului Ax=b calculata de noi
    :param d: diagonala principala din matricea initiala
    :param b: membru drept din sistemului Ax=b
    :return: norma euclidiana pentru |Ax-b|
    """
    n = len(a)
    y = [0 for _ in range(n)]
    z = 0
    for i in range(n):
        for j in range(n):
            if i < j:  # partea de sus a matricei
                y[i] += a[i][j] * x[j]
            if i == j:
                y[i] += d[j] * x[j]
            if i > j:
                y[i] += a[j][i] * x[j]
        y[i] -= b[i]
        z += y[i] ** 2
    return math.sqrt(z)


def inverse_matrix(a, epsilon):
    n = len(a)
    a_inverse = [[0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        b = [0 for _ in range(n)]
        b[j] = 1
        y = forward_substitution(a, b, epsilon)
        x = backward_substitution(a, y, epsilon)
        for i in range(n):
            a_inverse[i][j] = x[i]
    return a_inverse


if __name__ == '__main__':
    # n, epsilon, a, b = read_data_from_file("data.txt")
    # n, epsilon, a, b = read_data_from_console()
    n, epsilon, a, b = generate_random_matrix()
    a_np = np.array(a)
    d = [a[i][i] for i in range(n)]  # diagonala initiala
    cholesky_decomposition(a, n, epsilon)
    print("Determinant A: ", determinant(a))
    # b = [16, 27, 41]
    b_np = np.array(b)
    x = solve_eq(a, b, epsilon)
    print("Solutia sistemului Ax=b: ", x)
    print("Norma euclidiana:", norm(a, x, d, b))

    L_np = np.linalg.cholesky(a_np)
    print("Descompunere Cholesky numpy:\n", L_np)
    x_np = np.linalg.solve(a_np, b_np)
    print("Solutia sistemului Ax=b calculat de numpy: ", x_np)

    a_inverse = inverse_matrix(a, epsilon)
    a_inverse_np = np.linalg.inv(a_np)
    print("Inversa matricei A:", a_inverse)
    print("Inversa matricei A cu numpy:\n", a_inverse_np)
    print(np.linalg.norm(a_inverse - a_inverse_np, ord="fro"))
