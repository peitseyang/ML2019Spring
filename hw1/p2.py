import numpy as np
import matplotlib.pyplot as plt
import copy

data = [line.strip('\n').split('\t') for line in open('./data2.txt','r')]
x = np.vstack(data)[:,0:4]
x = np.concatenate((np.ones((x.shape[0], 1)), x), 1).astype(np.float32)
y = np.vstack(data)[:,4].astype(np.float32)

iter_num = 15000000
lr = 0.0000002
init_w = np.zeros(5).astype(np.float32)
print('Initialize w as [ %s ],\n running %d iteration and the learning rate is %.7f' 
    % (', '.join(str(w) for w in init_w), iter_num, lr))

for i in range(iter_num):
    loss = np.dot(x, init_w) - y
    grad_w = 2.0 * np.dot(x.T, loss)
    init_w -= lr * grad_w

    if i % 500 == 499:
        print('[%d/%d] loss = %.4f\r'
            % (i+1, iter_num, np.mean(np.square(loss))), end='')
print('Result: Weight (w0, w1, w2, w3, w4) = (%s), loss = %.4f' 
    % (', '.join(str(round(w, 4)) for w in init_w), np.mean(np.square(loss))))

# cheat_ans = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
for i in [[6.8, 210, 0.402, 0.739], [6.1, 180, 0.415, 0.713]]:
    _x = copy.deepcopy(i)
    _x.insert(0, 1.)
    print('When (x1, x2, x3, x4) = %s, the predicted value will be %.1f' 
        % (', '.join(str(round(_i, 4)) for _i in i), np.dot(_x, init_w)))
