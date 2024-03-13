from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_jobLogFromFile(result_path,indexCol):
    filename = "../Results/" + result_path
    print(filename)
    df = pd.read_csv(filename, index_col=indexCol,
                     sep="\s+|;+|\s+;+|;+\s+",
                     header = None,
                     names=['id', "autoAllocNode", 'estRunTime', 'run','wait', 'submit',
                            'start', 'end','qosRunTime'],
                     engine='python')

    return df
def get_jobLogOriginFromFile(logPath,indexCol):
    filename = logPath
    print(filename)
    df = pd.read_csv(filename, index_col=indexCol,
                     sep="\s+|\t+|\s+\t+|\t+\s+", comment=';',
                     header = None,
                     names=['id', "SubmitTime", 'WaitTime', 'RunTime','TaskCount', 'CPUTime',
                            'UsedMEM', 'TaskReq',  'ReqWallTime', 'RequestedMemory',
                            'Status', 'User', 'Group', 'Exe', 'Class', 'Partition',
                            'prejob', 'qosRunTime'],
                     engine='python')

    return df
def analysisInputJobLog(result_path):
    df = get_jobLogOriginFromFile(result_path,'id')
    print('test')


def analysisResult(result_path):
    df = get_jobLogFromFile(result_path,0)

    # 计算平均等待时间
    Avewt = df['wait'].mean()
    print('平均等待时间:', Avewt)

    # 计算平均有界减速度
    df['wait'] = df['wait'].astype(float)
    df['run'] = df['run'].astype(float)
    Span_List = (df['wait'] + df['run']).to_list()
    Const_List = pd.Series(10, df['wait'].index).to_list()
    RunTime_List = df['run'].to_list()
    Bsld_List = []
    for index in range(0, len(RunTime_List)):
        Bsld = max((Span_List[index] / (max(RunTime_List[index], Const_List[index]))), 1)
        Bsld_List.append(Bsld)
    AveBsld = np.mean(Bsld_List)
    print('平均有界减速度:', AveBsld)

    # 统计作业QoS满足率
    df_len = len(df)
    qos_num = 0
    for index,row in df.iterrows():
        if row['qosRunTime'] >= (row['wait']+row['run']):
            qos_num += 1
    print('QoS满足率:',qos_num / df_len)




def analysisReward(result_path):
    filename = "../Results/" + result_path
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 将每行的数字存入一个列表
    numbers_list = [float(line.strip()) for line in lines]

    # 创建DataFrame
    df = pd.DataFrame(numbers_list, columns=['Reward'])

    # 绘制折线统计图
    plt.plot(df['Reward'])
    plt.xlabel('Index')
    plt.ylabel('Reward')
    plt.show()

    print('系统平均Reward:', df['Reward'].mean())
def analysisCoresSys(result_path):
    filename = "../Results/" + result_path
    numbers_list = []
    with open(filename, 'r') as file:
        for line in file:
            # 将每一行按照;分割成两个数字
            numbers = line.strip().split(';')
            # 将第二个数字添加到列表中
            numbers_list.append(float(numbers[1]))

    df = pd.DataFrame({'coresSys': numbers_list})

    # 绘制折线统计图
    plt.plot(df['coresSys'])
    plt.xlabel('Index')
    plt.ylabel('coresSys')
    plt.show()

    print('系统平均资源率:',df['coresSys'].mean())
if __name__ == '__main__':
    analysisResult("ANL_log_data14_13_02.rst")
    analysisReward("ANL_log_data14_13_02.rwd")
    analysisCoresSys("ANL_log_data14_13_02.ult")