def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for i in range(steps):
        x0 = x0-lr*quadratic_derivative(a,b,c,x0)
    return float(x0)
def quadratic_derivative(a,b,c,x0):
    return 2*a*x0+b
    
    pass