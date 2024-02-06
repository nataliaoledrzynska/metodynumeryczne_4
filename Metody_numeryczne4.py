import numpy as np 
import matplotlib.pyplot as plt

def f(seir: list, gamma, beta, sigma):
    S = seir[0]
    E = seir[1]
    I = seir[2]

    ds = -(beta * I * S)
    de = (beta * I * S) - sigma * E
    di = sigma * E - gamma * I
    dr = gamma * I

    return np.array([ds, de, di, dr])

def Runge_Kutta_4(seir, dt, gamma, sigma, beta):
    f1 = dt * f(seir, gamma, beta, sigma)
    f2 = dt * f(seir + f1 / 2., gamma, beta, sigma)
    f3 = dt * f(seir + f2 / 2., gamma, beta, sigma)
    f4 = dt * f(seir + f3, gamma, beta, sigma)

    next_x = seir + 1/6 * (f1 + 2 * f2 + 2 * f3 + f4)
    return next_x

def SEIR_model(gamma, sigma, beta):
    t0 = 0
    te = 50
    dt = .01
    time = np.arange(t0, te, dt)

    seir = np.zeros([time.shape[0], 4])
    seir[0, 0] = 0.99
    seir[0, 1] = 0.01
    seir[0, 2] = 0.
    seir[0, 3] = 0.

    for i in range(time.shape[0] - 1):
        seir[i+1] = Runge_Kutta_4(seir[i], dt, gamma, sigma, beta)

    t = time.shape[0]
    S = [seir[j, 0] for j in range(t)]
    E = [seir[j, 1] for j in range(t)]
    I = [seir[j, 2] for j in range(t)]
    R = [seir[j, 3] for j in range(t)]

    plt.plot(time, S, label='Zdrowi')
    plt.plot(time, E, label='Chorzy niezarażający')
    plt.plot(time, I, label='Chorzy zarażający')
    plt.plot(time, R, label='Ozdrowieni/Martwi')
    plt.ylabel('Czas')
    plt.xlabel('Znormalizowana populacja')
    plt.legend()
    plt.show()

def main():
    SEIR_model(0.1, 1, 1)   # zadanie 1
    SEIR_model(0.1, 1, 0.5) # zadanie 2
    SEIR_model(0.4, 1, 4)   # zadanie 3
    SEIR_model(1, 1, 0.3)   # zadanie 3


main() 