# 该类给定一个计算作业，使用amdahl定律预测不同结点下的运行时间，返回满足QoS需求的最低需要结点数
from .varyCoresRunTime import *

def getRightNodes(job_info,cores_OneNode,idle_nodes):
    rightNodes = 1
    for node in range(1,idle_nodes+1):
        length = amdahlSpeedUp(job_info['run'],job_info['usedProc'],cores_OneNode*node)
        if job_info['qosRunTime'] >= length:
            rightNodes = node
            break
    return rightNodes

