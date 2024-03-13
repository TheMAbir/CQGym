from src_fc.CqSim.Cqsim_sim import Cqsim_sim
from gym import Env, spaces
import numpy as np
from src_fc.CqGym.GymState import GymState
from src_fc.CqGym.GymGraphics import GymGraphics
from copy import deepcopy

class CqsimEnv(Env):

    def __init__(self, module, debug=None, job_cols=0, window_size=0, do_render=False, render_interval=1, render_pause=0.01,alg_str=''):
        Env.__init__(self)  #  代表强化学习环境,它定义了强化学习算法与环境之间的交互接口.自定义的强化学习环境需要继承 Env 类,并实现其中的一些方法,如 step(),reset(),render()等，以提供与环境的交互.

        # Maintaining Variables for reset.
        module['info'].alg_str = alg_str
        self.simulator_module = module
        self.simulator_debug = debug

        # Initializing CQSim Backend
        self.simulator = Cqsim_sim(module, debug=debug)
        self.simulator.start()
        # Let Simulator load completely.
        self.simulator.pause_producer()

        GymState._job_cols_ = job_cols
        GymState._window_size_ = window_size
        self.gym_state = GymState()


        GymState._cores_node_ = module['node'].nodeStruc[0]['proc']
        # Defining Action Space and Observation Space.
        self.action_space = spaces.Discrete(window_size)
        self.observation_space = spaces.Box(shape=(1, self.simulator.module['node'].get_tot() +
                                                   window_size * job_cols, 2),
                                            dtype=np.float32, low=0.0, high=1000000.0)

        # Define object for Graph Visualization:
        self.graphics = GymGraphics(do_render, render_interval, render_pause)
        self.rewards = []
        self.rewards_info = []
        self.iter = 0

    def reset(self):
        """
        Reset the Gym Environment and the Simulator to a fresh start.
        :return: None
        """
        del self.simulator
        self.simulator = Cqsim_sim(deepcopy(self.simulator_module), debug=self.simulator_debug)
        self.simulator.start()
        # Let Simulator load completely.
        self.simulator.pause_producer()

        # Reinitialize Local variables
        self.gym_state = GymState()
        self.graphics.reset()
        self.rewards = []
        self.iter = 0

    def render(self, mode='human'):
        """
        :param mode: [str] :- No significance in the current version, only maintained to adhere to OpenAI-Gym standards.
        :return: None
        """
        # Show graphics at intervals.
        self.graphics.visualize_data(self.iter, self.gym_state, self.rewards)

    def get_state(self):
        """
        This function creates GymState Object for maintaining the current state of the Simulator.
        :return: [GymState]
        """
        self.gym_state = GymState()
        self.gym_state.define_state(self.simulator.currentTime,  # Current time in the simulator.
                                    self.simulator.simulator_wait_que_indices,  # Current Wait Queue in focus.
                                    self.simulator.module['job'].job_info(-1),  # All the JobInfo Dict.
                                    self.simulator.module['node'].nodeStruc,  # All the NodeStruct Dict.
                                    self.simulator.module['node'].get_idle())  # Number of Nodes available.
        return self.gym_state

    def step(self, action: int):
        """
        :param action: [int] :- Wait-Queue index of the selected Job.
                                Note - this is not Job Index.
        :return:
        gym_state: [GymState]   :- Contains all the information for the next state.
                                   gym_state.feature_vector stores Feature vector for the current state.
        done: [boolean]         :- True - If the simulation is complete.
        reward : [float]        :- reward for the current action.
        """
        self.iter += 1
        ind_alloc = action
        self.simulator.simulator_cpu_count = ind_alloc + 1  # PPO算法中的ActorNet网络的资源分配输出为0~(alloc_outputs-1),当+1后,应为1~alloc_outputs
        if action < 0:  # 当分配的结点数小于0，说明此时系统空闲资源数不满足最低的分配限度
            self.simulator.pause_producer()
        # print( "RL智能体资源分配数量的选择为:",self.simulator.simulator_cpu_count)

        reward = self.gym_state.get_reward(self.simulator.simulator_cpu_count,self.simulator.selectJobInfo)

        # Maintaining data for GymGraphics
        self.rewards.append(reward)

        # 将获得reward时,系统状态保存(等待队列中作业数量),可用资源数量,系统选择的资源数量
        self.simulator.debug.debug("wait_queue_size: {} Avail Resource: {} Auto Alloc Nodes: {}".format(len(self.simulator.module['job'].job_wait_list), str(self.simulator.module['node'].avail), str(self.simulator.simulator_cpu_count)), 3)

        # Gym Paused, Running simulator.(Gym环境是生产者,而simlator是消费者)   上述代码已经完成了等待队列中job的选择,完成了action动作,应启动simulator线程,此线程完成调度器的模拟
        self.simulator.pause_producer()     #当完成在等待队列中选择job后，要将模拟权限交给模拟调度系统，故再次执行pause_producer

        # Simulator executed with selected action. Retrieving new State.(刚启动了调度系统的模拟,执行了所选择的动作,根据是否模拟结束选择下一个状态)
        if self.simulator.is_simulation_complete:
            # Return an empty state if the Simulation is complete. Avoids NullPointer Exceptions.
            self.gym_state = GymState()
        else:
            self.get_state()

        return self.gym_state, self.simulator.is_simulation_complete, reward

    def step_forRigid(self, action: int):
        """
        :param action: [int] :- Wait-Queue index of the selected Job.
                                Note - this is not Job Index.
        :return:
        gym_state: [GymState]   :- Contains all the information for the next state.
                                   gym_state.feature_vector stores Feature vector for the current state.
        done: [boolean]         :- True - If the simulation is complete.
        reward : [float]        :- reward for the current action.
        """
        self.iter += 1
        ind = action
        print("Wait Queue at Step Func - ", self.simulator.simulator_wait_que_indices)
        self.simulator.simulator_wait_que_indices = [self.simulator.simulator_wait_que_indices[ind]] + \
                                                     self.simulator.simulator_wait_que_indices[:ind] + \
                                                     self.simulator.simulator_wait_que_indices[ind + 1:]

        # reward = self.gym_state.get_reward(self.simulator.simulator_wait_que_indices[0])

        # Maintaining data for GymGraphics
        # self.rewards.append(reward)

        # Gym Paused, Running simulator.(Gym环境是生产者,而simlator是消费者)   上述代码已经完成了等待队列中job的选择,完成了action动作,应启动simulator线程,此线程完成调度器的模拟
        self.simulator.pause_producer()     #当完成在等待队列中选择job后，要将模拟权限交给模拟调度系统，故再次执行pause_producer

        # Simulator executed with selected action. Retrieving new State.(刚启动了调度系统的模拟,执行了所选择的动作,根据是否模拟结束选择下一个状态)
        if self.simulator.is_simulation_complete:
            # Return an empty state if the Simulation is complete. Avoids NullPointer Exceptions.
            self.gym_state = GymState()
        else:
            self.get_state()

        return self.gym_state, self.simulator.is_simulation_complete