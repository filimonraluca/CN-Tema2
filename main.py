import math


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
    d = [a[i][i] for i in range(n)]
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
    print(d)


def determinant(a):
    det = 1
    for i in range(n):
        det *= a[i][i]
    return det ** 2


if __name__ == '__main__':
    n, epsilon, a, b = read_data_from_file("data.txt")
    cholesky_decomposition(a, n)
    print(determinant(a))
