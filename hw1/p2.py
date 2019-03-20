import numpy as np
import matplotlib.pyplot as plt
import copy

data = [line.strip('\n').split('\t') for line in open('./data2.txt','r')]
x1, x2, x3, x4, y = zip(*data)
x, y = np.array([[1., x1[i], x2[i], x3[i], x4[i]] for i in range(len(x1))]).astype(float), list(map(float, y))
# x1, x2, x3, x4, y = list(map(float, x1)), list(map(int, x2)), list(map(float, x3)), list(map(float, x4)), list(map(float, y))

iter_num = 20000
lr = 0.000022089
init_w = np.array([0.0 for i in range(5)])
print('Initialize w as [ %s ],\n running %d iteration and the learning rate is %f' 
    % (', '.join(str(w) for w in init_w), iter_num, lr))
def f(_X): return np.dot(_X, init_w)

for i in range(iter_num):
    grad_w = np.array([0.0 for i in range(5)])
    err = 0.0
    for j in range(len(data)):
        temp = y[j] - f(x[j])
        for k in range(len(grad_w)):
            if k == 0:
                grad_w[k] -= (2 / len(data)) * temp
            else:
                grad_w[k] -= (2 / len(data)) * temp * x[j][k]
        err += temp ** 2
    for l in range(len(grad_w)):
        init_w[l] -= lr * grad_w[l]
    err /= len(data)
    if i % 500 == 499:
        print('[%d/%d] W(w0, w1, w2, w3, w4) = (%s), Err = %.4f'
            % (i+1, iter_num, ', '.join(str(round(w, 4)) for w in init_w), err))
print('Result: Weight (w0, w1, w2, w3, w4) = (%s)' % (', '.join(str(round(w, 4)) for w in init_w)))

for i in [[6.8, 210, 0.402, 0.739], [6.1, 180, 0.415, 0.713]]:
    _x = copy.copy(i)
    _x.insert(0, 1.)
    print('When (x1, x2, x3, x4) = %s, the predicted value will be %.1f' 
        % (', '.join(str(round(_i, 4)) for _i in i), f(_x)))
