import math
import os

import numpy as np
import random

global from_file, in_file, generate, read_file, write_file


def read_data_from_file():
    global read_file
    try:
        f = open(read_file, "r")
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


def write_matrix_in_file(a, n, text):
    global write_file
    try:
        f = open(write_file, "a")
        f.write(text + '\n')
        for i in range(n):
            for j in range(n):
                f.write(str(a[i][j]) + " ")
            f.write('\n')
        f.close()
    except OSError:
        print("Error while writing data!")


def write_array_in_file(b, n, text):
    global write_file
    try:
        f = open(write_file, "a")
        f.write(text + '\n')
        for i in range(n):
            f.write(str(b[i]) + " ")
        f.write('\n')
        f.close()
    except OSError:
        print("Error while writing data!")


def write_value_in_file(x, text):
    global write_file
    try:
        f = open(write_file, "a")
        f.write(text + str(x) + '\n')
        f.close()
    except OSError:
        print("Error while writing data!")


def write_matrix_in_console(a, n):
    for i in range(n):
        for j in range(n):
            print(a[i][j], end=" ")
        print()


def write_array_in_console(b, n):
    for i in range(n):
        print(b[i], end=" ")
    print()


def generate_random_matrix():
    global in_file
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
    if in_file:
        write_matrix_in_file(a, n, "Matricea A generata:")
        write_array_in_file(b, n, "Vectorul termenilor liberi:")
    else:
        write_matrix_in_console(a, n)
        write_array_in_console(b, n)
    return n, epsilon, a, b


def cholesky_decomposition(a, n, epsilon):
    global in_file
    for i in range(n):
        for j in range(0, i + 1):
            if i == j:
                s = 0
                for k in range(i):
                    s += a[i][k] ** 2
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
    if in_file:
        write_matrix_in_file(a, n, "Descompunere Cholesky: ")
    else:
        print("Descompunere Cholesky: ")
        write_matrix_in_console(a, n)


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
    global in_file
    y = forward_substitution(a, b, epsilon)
    x = backward_substitution(a, y, epsilon)
    if in_file:
        write_array_in_file(y, n, "Solutia ecuatiei Ly=b:")
        write_array_in_file(x, n, "Solutia sistemului Ax=b: ")
    else:
        print("Solutia ecuatiei Ly=b:")
        write_array_in_console(y, n)
        print("Solutia sistemului Ax=b: ")
        write_array_in_console(x, n)
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
    in_file = True
    from_file = False
    read_file = "in_data.txt"
    write_file = "out_data.txt"
    generate = True
    if os.path.exists(write_file):
        os.remove(write_file)
    if from_file:
        n, epsilon, a, b = read_data_from_file()
    elif generate:
        n, epsilon, a, b = generate_random_matrix()
    else:
        n, epsilon, a, b = read_data_from_console()
    a_np = np.array(a)
    d = [a[i][i] for i in range(n)]  # diagonala initiala
    cholesky_decomposition(a, n, epsilon)
    b_np = np.array(b)
    x = solve_eq(a, b, epsilon)
    a_inverse = inverse_matrix(a, epsilon)
    a_inverse_np = np.linalg.inv(a_np)
    L_np = np.linalg.cholesky(a_np)
    x_np = np.linalg.solve(a_np, b_np)
    norma = norm(a, x, d, b)
    if not in_file:
        print("Determinant A: ", determinant(a))
        print("Norma euclidiana:", norma)
        print("Descompunere Cholesky numpy:\n", L_np)
        print("Solutia sistemului Ax=b calculat de numpy: ", x_np)
        print("Inversa matricei A:", a_inverse)
        print("Inversa matricei A cu numpy:\n", a_inverse_np)
        print("Norma pentru inversa: ", np.linalg.norm(a_inverse - a_inverse_np, ord="fro"))
    else:
        write_value_in_file(determinant(a), "Determinant A:")
        write_value_in_file(norma, "Norma euclidiana:")
        write_matrix_in_file(a_inverse,n, "Inversa matricei A:")
        write_value_in_file(np.linalg.norm(a_inverse - a_inverse_np, ord="fro"), "Norma pentru inversa: ")
