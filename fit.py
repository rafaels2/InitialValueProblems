import matplotlib.pyplot as plt
import scipy.optimize


def linear(x, *params):
    a, b = params
    return a * x + b


def fit_linear(x_values, y_values, title):
    print("Do fit")
    (a, b), _ = scipy.optimize.curve_fit(linear, x_values, y_values, p0=[2, 1])
    plt.figure()
    plt.title(f"Error rates of {title}; slope %.3f" % a)
    plt.scatter(x_values, y_values, marker="*")
    plt.plot(x_values, linear(x_values, a, b))
    plt.xlabel("$log_{10}(N)$")
    plt.ylabel("$log_{10}($error$)$")
    plt.show()
