import math
import numpy as np


def read_data_from_file(file):
    try:
        f = open(file, "r")
        n = int(f.readline())
        epsilon = pow(10, int(f.readline()))
        a = []
        for i in range(n):
            line = [float(x) for x in f.readline().split(" ")]
            a.append(line)
        b = [float(x) for x in f.readline().split(" ")]
        return n, epsilon, a, b
    except OSError:
        print("Error while reading data!")


def cholesky_decomposition(a, n):
    for i in range(n):
        for j in range(0, i + 1):
            if i == j:
                s = 0
                for k in range(i):
                    s += a[i][k] ** 2
                a[i][j] = math.sqrt(a[i][j] - s)
            else:
                s = 0
                for k in range(j):
                    s += a[i][k] * a[j][k]
                a[i][j] = (a[i][j] - s) / a[j][j]
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


def forward_substitution(L, b):
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
        y[i] = (b[i] - s) / L[i][i]
    return y


def backward_substitution(L, y):
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
        x[i] = (y[i] - s) / L[i][i]
    return x


def solve_eq(a, b):
    y = forward_substitution(a, b)
    print("Solutia ecuatiei Ly=b:", y)
    x = backward_substitution(a, y)
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


if __name__ == '__main__':
    n, epsilon, a, b = read_data_from_file("data.txt")
    a_np = np.array(a)
    d = [a[i][i] for i in range(n)]  # diagonala initiala
    cholesky_decomposition(a, n)
    print("Determinant A: ", determinant(a))
    # b = [16, 27, 41]
    b_np = np.array(b)
    x = solve_eq(a, b)
    print("Solutia sistemului Ax=b: ", x)
    print("Norma euclidiana:", norm(a, x, d, b))

    L_np = np.linalg.cholesky(a_np)
    print("Descompunere Cholesky numpy:\n", L_np)
    x_np = np.linalg.solve(a_np, b_np)
    print("Solutia sistemului Ax=b calculat de numpy: ", x_np)
