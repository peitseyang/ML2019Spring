import numpy as np
import matplotlib.pyplot as plt

data = [line.strip('\n').split('\t') for line in open('./data1.txt','r')]
x, y = zip(*data)
x, y = list(map(int, x)), list(map(float, y))

iter_num = 100
lr = 0.001
init_w0 = 0.0
init_w1 = 0.0
print('Initialize (w0, w1) as (%d, %d), running %d iteration and the learning rate is %f' 
    % (init_w0, init_w1, iter_num, lr))
def f(_x): return init_w0 + init_w1 * _x

for i in range(iter_num):
    grad_w0 = 0.0
    grad_w1 = 0.0
    err = 0.0
    for j in range(len(data)):
        grad_w0 -= (2 / len(data)) * (y[j] - f(x[j]))
        grad_w1 -= (2 / len(data)) * (y[j] - f(x[j])) * init_w0
        err += (y[j] - f(x[j])) ** 2
    init_w0 -= lr * grad_w0
    init_w1 -= lr * grad_w1
    err /= len(data)
    # print('[%d/%d] W(w0, w1) = (%d, %d), Err = %d'
    #     % (i+1, iter_num, round(init_w0, 2), round(init_w1, 2), round(err, 2)))
print('Result: Weight (w0, w1) = (%d, %d)' % (init_w0, init_w1))

for i in [45, 25]:
    print('When x = %d, the predicted value will be %d' % (i, f(i)))

x_range = np.linspace(min(x), max(x), 50)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_range, f(x_range), color='red', linewidth=2.0)
plt.scatter(x, y)
plt.show()