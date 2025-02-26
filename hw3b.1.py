import math

def gamma_function(x):
    if x == 1:
        return 1
    elif x == 0.5:
        return math.sqrt(math.pi)
    return (x - 1) * gamma_function(x - 1)

def km_value(m):
    return gamma_function((m + 1) / 2) / (math.sqrt(m * math.pi) * gamma_function(m / 2))

def t_distribution(m, z):
    km = km_value(m)
    integral_value = 0
    step = 0.001
    u = -z
    while u <= z:
        integral_value += (1 + (u**2 / m))**(-(m + 1) / 2) * step
        u += step
    return km * integral_value

def main(m, z):
    if m <= 0:
        raise ValueError("Degrees of freedom must be positive.")
    probability = t_distribution(m, z)
    return probability

if __name__ == "__main__":
    test_cases = [(7, 1.0), (11, 2.0), (15, 2.5)]
    for m, z in test_cases:
        try:
            result = main(m, z)
            print(f"Computed probability F(z) for m={m} and z={z}: {result:.5f}")
        except ValueError as e:
            print(f"Invalid input for m={m}: {e}")
