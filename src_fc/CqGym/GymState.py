from src_fc.RunTime.varyCoresRunTime import *

class GymState:

    _job_cols_ = 2
    _window_size_ = 1
    # 每个结点所对应的核数
    _cores_node_ = 32
    def __init__(self):
        # Variable to maintain the info received
        self.current_time = None
        self.wait_que = None
        self.wait_que_size = 0
        self.job_info = {}

        self.job_vector = []
        self.node_vector = []
        self.total_nodes = 0
        self.idle_nodes = 0
        self.feature_vector = []

    def define_state(self, current_time, wait_que_indices, job_info_dict, node_info_list, idle_nodes_count):
        """
        :param wait_que_indices: List[Integer] - indices of the jobs in wait que, List size limited.
        :param job_info_dict: Dict{Integer: Info} - Information of all the jobs from simulator
        :param node_info_list: List[Node Info] - Information of all the nodes from simulator
        :return: State parsable by the RL Model in use - Eg. Numpy Array
        """
        self.current_time = current_time
        self.wait_que = wait_que_indices[:]
        self.wait_que_size = len(self.wait_que)
        self.job_info = job_info_dict
        self.total_nodes = len(node_info_list)
        self.idle_nodes = idle_nodes_count

        self.wait_job = [job_info_dict[ind] for ind in wait_que_indices]

        wait_job_input = self.preprocessing_queued_jobs(
            self.wait_job, current_time)

        # 将作业状态与系统状态合并 20表示将系统资源可用情况复制20份
        self.feature_vector = self.make_feature_vector(wait_job_input,20)

        def vector_reshape(vec):
            return vec.reshape(tuple([1]) + vec.shape)
        self.feature_vector = vector_reshape(self.feature_vector)



    def preprocessing_queued_jobs(self, wait_job, currentTime):
        job_info_list = []
        deadLine = wait_job[0]['qosRunTime']
        jobQueueTime = self.current_time - wait_job[0]['submit']
        info = [[deadLine,jobQueueTime]]
        # 加入等待队列中作业数量信息
        info.append([len(wait_job),len(wait_job)])
        # 加入此作业在其他结点数下的运行时间信息
        perStageSize = int(self.total_nodes / 10)
        for index in range(perStageSize):
            info.append([index+1,amdahlSpeedUp(wait_job[0]['run'],  wait_job[0]['usedProc'],
                                                           GymState._cores_node_ * (index+1))])
        job_info_list.append(info)


        return job_info_list

    def preprocessing_system_status(self, node_struc, currentTime):
        node_info_list = []
        # Each element format - [Availbility, time to be available] [1, 0] - Node is available
        for node in node_struc:
            info = []
            # 当结点可用时,node_struct中每个结点的state为-1,小于0
            # 当结点不可用时,node_struct中每个结点的state大于等于0
            # avabile 1, not available 0
            if node['state'] < 0:
                # 1表示结点可用
                info.append(1)
                info.append(0)
            else:
                # 0表示结点不可用,预计结点的可获得时间
                info.append(0)
                info.append(node['end'] - currentTime)
                # Next available node time.

            node_info_list.append(info)
        return node_info_list

    def make_feature_vector_twoStateMake(self, jobs, system_status):
        # Remove hard coded part !
        job_cols = self._job_cols_
        window_size = self._window_size_
        input_dim = [len(system_status) + window_size *
                     job_cols, len(system_status[0])]

        fv = np.zeros((1, input_dim[0], input_dim[1]))
        i = 0
        for idx, job in enumerate(jobs):
            fv[0, idx * job_cols:(idx + 1) * job_cols, :] = job
            i += 1
            if i == window_size:
                break
        fv[0, job_cols * window_size:, :] = system_status
        return fv


    def make_feature_vector(self, jobs,sys_size):
        # Remove hard coded part !
        job_cols = self._job_cols_
        window_size = self._window_size_
        diffNodesInfo = int(self.total_nodes / 10)
        # input dim的行数包括三方面信息: 1行表示qosRunTime，jobQueueTime 2行表示 等待队列中作业数量,等待队列中作业数量 3行开始不同结点数下运行时间 再加 系统中剩余节点关系
        input_dim = [window_size *
                     job_cols+diffNodesInfo+sys_size, job_cols]

        fv = np.zeros((1, input_dim[0], input_dim[1]))
        i = 0
        for idx, job in enumerate(jobs):
            fv[0, idx * job_cols:(idx + window_size *
                     job_cols+diffNodesInfo), :] = job
            i += 1
            if i == window_size:
                break
        fv[0, window_size * job_cols+diffNodesInfo:, :] = self.idle_nodes
        return fv


    def get_max_wait_time_in_queue(self):
        job_cnt = 0
        max_wait_time_in_que = 0
        max_job_size_in_que = 0
        total_wait_time = 0
        total_wait_core_seconds = 0
        for job_id in self.job_info:
            job_cnt += 1
            job = self.job_info[job_id]
            if job_cnt <= self._window_size_:
                max_wait_time_in_que = max(
                    max_wait_time_in_que, self.current_time - job['submit'])
                max_job_size_in_que = max(max_job_size_in_que, job['autoAllocNode'])
            total_wait_time += job['estRunTime']
            total_wait_core_seconds += job['estRunTime'] * job['autoAllocNode']
        return max_wait_time_in_que, max_job_size_in_que, total_wait_time, total_wait_core_seconds, job_cnt
    #   该reward函数的设计从处理大规模计算作业与优化资源利用率出发
    def get_reward(self, auto_cpu_count,selectJobInfo):
        tmp_reward = 0

        selectJobInfo['autoAllocNode'] = auto_cpu_count
        selectJobInfo['run'] = amdahlSpeedUp(selectJobInfo['run'],  selectJobInfo['usedProc'],
                                                           GymState._cores_node_ * selectJobInfo['autoAllocNode'])
        selectJobInfo['estRunTime'] = amdahlSpeedUp(selectJobInfo['estRunTime'],
                                                                  selectJobInfo['usedProc'],
                                                                  GymState._cores_node_ * selectJobInfo[
                                                                      'autoAllocNode'])
        selectJobInfo['usedProc'] = GymState._cores_node_ * selectJobInfo['autoAllocNode']
        if auto_cpu_count <= self.idle_nodes:
            qosRunTime = selectJobInfo['qosRunTime']
            makespan = selectJobInfo['run'] + self.current_time - selectJobInfo['submit']
            if makespan <= qosRunTime:
                # tmp_reward += ((qosRunTime - makespan) / makespan)
                if makespan == 0 or qosRunTime == 0:
                    tmp_reward = 1
                else:
                    tmp_reward += qosRunTime / makespan
            else:
                # tmp_reward += - ((makespan - qosRunTime) / qosRunTime)
                tmp_reward += - (makespan / qosRunTime)
        else:
            # if self.idle_nodes > 0:
            #     tmp_reward += -((auto_cpu_count - self.idle_nodes) / self.idle_nodes)
            # else:
            #     tmp_reward += -1
            tmp_reward += -2

        return tmp_reward
