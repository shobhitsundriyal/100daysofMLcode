#import
#least sum of squared value
def calculate_error(b, m, data):
    total_error = 0
    for i in range(0, len(data)):
        x = points[i,0]
        y = points[i,1]
        total_error += (y - (m*x + b)) ** 2
    total_error = total_error / i
    return total_error
