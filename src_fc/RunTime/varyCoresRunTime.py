import numpy as np

# 使用加速比模型计算不同核数下的运行时间
def getOriginData(length,proc_num):
    amdahl_a = np.random.uniform(0,10)
    amdahl_v = np.random.randint(2,8)

    # amdahl_sequence = amdahl_a / (10**amdahl_v)
    amdahl_sequence = 5e-4

    comm_a = np.random.uniform(1,2)
    comm_v = np.random.randint(0,4)

    comm_num = comm_a * (2**comm_v)

    # result = (proc_num * length - comm_num*proc_num*(proc_num-1)) / (1-amdahl_sequence + proc_num*amdahl_sequence)
    result = proc_num*length / (amdahl_sequence*proc_num + 1-amdahl_sequence)
    return result


def amdahlSpeedUp(length,proc_num,cores):
    seq = 5e-4
    seq_length = getOriginData(length,proc_num)
    result = (int)(seq_length * (seq + (1.0 - seq) / cores))
    return result

def limiteAmdahlSpeedUp(length,proc_num):
    seq = 5e-4
    seq_length = getOriginData(length, proc_num)
    result = (int)(seq_length * seq)
    return result