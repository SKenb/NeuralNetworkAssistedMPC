import numpy as np

def noReaction(Cs, i, theta): 
    return 0*Cs[i, :]

def dummyReaction(Cs, i, theta): 
    # C1, C2, C3
    # C1 + C2 --> C3
    assert Cs.shape[0] == 3

    theta_K = theta + 273.15
    rate = Cs[0, :] * Cs[1, :] * 10 * np.exp(-10e3 / theta_K / 8.314)
    return rate if i == 2 else -1 * rate

def paalKnorrReaction(Cs, i, theta, A1 = 1, A2 = 5, Ea1 = 15e3, Ea2 = 12e3, k11 = 1, k12 = 1, k21 = 1, k22 = 1):
    # C1, C2, C3, C4
    # C1 + C2 --> C3
    # C2 + C3 --> C4
    assert Cs.shape[0] == 4

    theta_K = theta + 273.15

    rate1 = Cs[0, :] * Cs[1, :] * A1 * np.exp(-Ea1 / theta_K / 8.314)
    rate2 = Cs[1, :] * Cs[2, :] * A2 * np.exp(-Ea2 / theta_K / 8.314)

    net = np.array([-k11*rate1, -k12*rate1-k21*rate2, rate1-k22*rate2, rate2])

    return net[i, :]
