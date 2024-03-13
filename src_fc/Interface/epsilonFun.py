import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_decay(initial_epsilon, decay_rate, current_step):
    epsilon =  sigmoid(current_step)
    return epsilon
def step_decay(initial_epsilon, decay_rate, step_size, current_step):
    num_steps = current_step // step_size  # 计算当前步数处于第几个阶梯
    epsilon = initial_epsilon * (decay_rate ** num_steps)  # 按照衰减率计算当前阶梯的epsilon值
    if epsilon < 0.1 :
        return 0.1
    return epsilon

if __name__=="__main__":
    for step in range(1,10000):
        print(step," ",step_decay(0.9,0.999,10,step))