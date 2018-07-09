from array import array
def gradient_descent(data, start_b, start_m, rate , iterations):
    b = start_b
    m= start_m
    for i in range(iterations):
        [b, m] = step_gradient(b, m, array(data), rate)
    return [b, m]

def step_gradient(current_b, current_m, data, rate):
    b_gradient = 0
    m_gradient = 0
    l = len(data)
    N = float(l)
    for i in range(l):
        x = data[i, 0]
        y = data[i, 1]
        b_gradient += -(2/N) * (y - (current_m * x) + current_b)
        m_gradient += -(2/N) * x * (y - (current_m * x) + current_b)
    new_b = current_b - (rate * b_gradient)
    new_m = current_m - (rate * m_gradient)
    return [new_b, new_m]
