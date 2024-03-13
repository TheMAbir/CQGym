from src_fc.CqGym.Gym import CqsimEnv
from src_fc.Models.PG import PG
import torch
import random
from src_fc.Interface.epsilonFun import *




perPartitionNode = 10
def get_action_from_output_vector(output_vector, sys_size, is_training,epsilon,sysIdleNodes):
    sysIdleNodes = int(sysIdleNodes / perPartitionNode)
    minAllocNodes = 1
    action_p = torch.softmax(
        output_vector[minAllocNodes:sysIdleNodes], dim=-1)
    action_p = action_p.detach().cpu()
    action_p = np.array(action_p)
    action_p /= action_p.sum()
    # 若minAllocNodes与sysIdleNodes相同,即截出来的列表中无元素. 返回0表示在第一个分区，即1-perPartitionNode个结点中选择结点
    if minAllocNodes == sysIdleNodes:
        return 0
    elif sysIdleNodes == 0:     # 若return -1说明当前可用结点数小于perPartitionNode,将所用可用资源作为智能分配结点数
        return -1
    if is_training and np.random.uniform() < epsilon:
        wait_queue_ind = np.random.choice(len(action_p), p=action_p)
    else:
        wait_queue_ind = np.argmax(action_p)
    return wait_queue_ind


def model_training(env, weights_file_name=None, is_training=False, output_file_name=None,
                   window_size=50, sys_size=0, learning_rate=0.1, gamma=0.99, batch_size=10, do_render=False, layer_size=[],on_cuda=None):
    use_cuda = torch.cuda.is_available()
    if use_cuda and not on_cuda:
        device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and on_cuda:
        device = torch.device(on_cuda)
    if use_cuda is False:
        device = torch.device("cuda" if use_cuda else "cpu")

    output_size = int(sys_size / perPartitionNode)
    num_inputs = window_size * 2 + output_size + 20  # 等待队列中作业表示为[2,2]的矩阵,包括job size,job ested runtime,priority,job queued time,每个作业都需要表示成[2,2],共有50个作业,即50*2;每个结点表示为[1,2],共有sys_size个结点;[100,2] + [4360,2] => [4460,2]
    pg = PG(env,num_inputs, output_size, learning_rate,
            gamma, batch_size, layer_size=layer_size,device = device)

    if weights_file_name:
        pg.load_using_model_name(weights_file_name)

    obs = env.get_state()
    done = False

    step_num = 1
    while not done:

        env.render()
        state = torch.FloatTensor(obs.feature_vector).to(device)
        probs = pg.act(state)
        action_p = torch.softmax(probs.detach(), dim=-1)
        epsilon = step_decay(0.9, 0.9999, 10, step_num)

        # 上述函数选择出的资源分配方案是所设定分类中的一个,需要再次完成随机选择.
        action = get_action_from_output_vector(
            probs.detach(), output_size, is_training,epsilon,env.simulator.module['node'].idle)
        one_PartitionSize = sys_size / output_size
        if action == -1:
            # 到action返回-1时表示 当前可用的结点数小于一个最小分区大小(即小于perPartitionNode)
            if env.simulator.module['node'].idle < env.simulator.minAutoAllocNodes:
                action = -10
            else:
                action = env.simulator.module['node'].idle - 1
        else:
            action = random.randint((action) * one_PartitionSize, (action + 1) * one_PartitionSize)
            if action < env.simulator.minAutoAllocNodes:
                # 只要能进此if说明 自动分配的结点数 小于 引入的最小分配结点数
                if env.simulator.minAutoAllocNodes <= env.simulator.module['node'].idle:
                    # 当可用结点数 大于 最低结点限度时，说明可以修正到 最低结点限度上(修正这个的后果就是 本来分配一个很小的节点范围理论上Value较小，现在修正后可能所得的Value)
                    action = env.simulator.minAutoAllocNodes
                else:
                    action = -10
        new_obs, done, reward = env.step(action)
        pg.remember(obs.feature_vector, action,
                    reward, new_obs.feature_vector)
        if is_training and not done:
            pg.train()
        obs = new_obs
        step_num += 1

    if is_training and output_file_name:
        pg.save_using_model_name(output_file_name)

    return pg.rewards_seq


def model_engine(module_list, module_debug, job_cols=0, window_size=0, sys_size=0,
                 is_training=False, weights_file=None, output_file=None, do_render=False, learning_rate=0.1, reward_discount=0.99, batch_size=10, layer_size=[],on_cuda=None,alg_str='PG'):
    """
   Execute the CqSim Simulator using OpenAi based Gym Environment with Scheduling implemented using DeepRL Engine.

    :param module_list: CQSim Module :- List of attributes for loading CqSim Simulator
    :param module_debug: Debug Module :- Module to manage debugging CqSim run.
    :param job_cols: [int] :- No. of attributes to define a job.
    :param window_size: [int] :- Size of the input window for the DeepLearning (RL) Model.
    :param is_training: [boolean] :- If the weights trained need to be saved.
    :param weights_file: [str] :- Existing Weights file path.
    :param output_file: [str] :- File path if the where the new weights will be saved.
    :return: None
    """
    cqsim_gym = CqsimEnv(module_list, module_debug,
                         job_cols, window_size, do_render,alg_str=alg_str)
    return model_training(cqsim_gym, window_size=window_size, sys_size=sys_size, is_training=is_training,
                weights_file_name=weights_file, output_file_name=output_file, learning_rate=learning_rate, gamma=reward_discount, batch_size=batch_size, layer_size=layer_size,on_cuda=on_cuda)
