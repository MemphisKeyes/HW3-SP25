import math

def normal_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def simpsons_rule(f, a, b, n=1000):
    if n % 2:
        n += 1
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        integral += 2 * f(a + i * h)
    return (h / 3) * integral

def cumulative_probability(c, mu=0, sigma=1):
    return simpsons_rule(lambda x: normal_pdf(x, mu, sigma), mu - 5 * sigma, c)

def secant_method(target_p, mu=0, sigma=1, tol=1e-5, max_iter=100):
    c0, c1 = mu - sigma, mu + sigma
    for _ in range(max_iter):
        f_c0 = cumulative_probability(c0, mu, sigma) - target_p
        f_c1 = cumulative_probability(c1, mu, sigma) - target_p
        if abs(f_c1) < tol:
            return c1
        c2 = c1 - f_c1 * (c1 - c0) / (f_c1 - f_c0)
        c0, c1 = c1, c2
    return c1

def main(choice, mu, sigma, value):
    if choice == 'P':
        c = value
        probability = cumulative_probability(c, mu, sigma)
        return f"P(X < {c} | mu={mu}, sigma={sigma}) = {probability:.5f}"
    elif choice == 'C':
        target_p = value
        c_value = secant_method(target_p, mu, sigma)
        return f"Value of c for P(X < c | mu={mu}, sigma={sigma}) = {target_p} is {c_value:.5f}"
    else:
        return "Invalid choice. Please enter 'P' or 'C'."

if __name__ == "__main__":
    test_cases = [
        ('P', 0, 1, 1),
        ('C', 0, 1, 0.84134),
        ('P', 5, 2, 6),
        ('C', 5, 2, 0.97725)
    ]
    for test in test_cases:
        print(main(*test))
