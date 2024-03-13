from src_fc.CqGym.Gym import CqsimEnv


def model_training(env, do_render=False):

    obs = env.get_state()
    done = False

    while not done:
        env.render()
        action = -1
        combined_score = 0
        knob = 0
        min_runTime = float('Inf')
        for i, v in enumerate(obs.wait_job):
            # 计算资源对齐评分
            tmp_align_score = v['autoAllocNode'] / obs.idle_nodes
            # 计算作业长度评分(作业长度越短,相应评分越高)
            tmp_sjf_score = 1 / float(v['estRunTime'])
            # Knob参数用来平衡 资源评分与作业长度评分
            tmp_combined_score = knob*tmp_align_score + (1-knob)*tmp_sjf_score

            if tmp_combined_score > combined_score:
                combined_score = tmp_combined_score
                action = i

        new_obs, done = env.step_forRigid(action)


def model_engine(module_list, module_debug, job_cols=0, window_size=0, sys_size=0, do_render=False,alg_str='SJF'):
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
    model_training(cqsim_gym)