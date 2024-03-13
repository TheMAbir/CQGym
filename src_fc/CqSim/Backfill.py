
__metaclass__ = type

class Backfill:
    def __init__(self, mode = 0, ad_mode = 0, node_module = None, debug = None, para_list = None):
        self.myInfo = "Backfill"
        self.mode = mode
        self.ad_mode = ad_mode
        self.node_module = node_module
        self.debug = debug
        self.para_list = para_list
        self.current_para = []
        self.wait_job = []

        self.debug.line(4," ")
        self.debug.line(4,"#")
        self.debug.debug("# "+self.myInfo,1)
        self.debug.line(4,"#")
        
    def reset (self, mode = None, ad_mode = None, node_module = None, debug = None, para_list = None):
        #self.debug.debug("* "+self.myInfo+" -- reset",5)
        if mode:
            self.mode = mode
        if ad_mode :
            self.ad_mode = ad_mode 
        if node_module:
            self.node_module = node_module
        if debug:
            self.debug = debug
        if para_list:
            self.para_list = para_list
        self.current_para = []
        self.wait_job = []
    
    def backfill (self, wait_job, para_in = None):
        #self.debug.debug("* "+self.myInfo+" -- backfill",5)
        if (len(wait_job) <= 1):
            return []
        self.current_para = para_in
        self.wait_job = wait_job
        job_list = self.main()
        return job_list
    
    def main(self):
        #self.debug.debug("* "+self.myInfo+" -- main",5)
        result = []
        if (self.mode == 1):
            # EASY backfill
            result = self.backfill_EASY()
        elif (self.mode == 2):
            # Conservative backfill
            result = self.backfill_cons()
        elif self.mode == 3:
            # Gym Based - RL Backfill
            result = self.backfill_RL()
        else:
            return None
        return result
    
    def backfill_EASY(self):
        #self.debug.debug("* "+self.myInfo+" -- backfill_EASY",5)
        backfill_list = []
        self.node_module.pre_reset(self.current_para['time'])
        self.node_module.reserve(self.wait_job[0]['proc'], self.wait_job[0]['index'], self.wait_job[0]['run'])
        i = 1
        job_num = len(self.wait_job)
        while (i < job_num):
            backfill_test = 0
            backfill_test = self.node_module.pre_avail(self.wait_job[i]['proc'], self.current_para['time'],
                                                       self.current_para['time'] + self.wait_job[i]['run'])
            if (backfill_test == 1):
                backfill_list.append(self.wait_job[i]['index'])
                self.node_module.reserve(self.wait_job[i]['proc'], self.wait_job[i]['index'], self.wait_job[i]['run'])
            i += 1
        return backfill_list
        
    def backfill_cons(self):
        #self.debug.debug("* "+self.myInfo+" -- backfill_cons",5)
        backfill_list=[]
        self.node_module.pre_reset(self.current_para['time'])
        self.node_module.reserve(self.wait_job[0]['proc'], self.wait_job[0]['index'], self.wait_job[0]['run'])
        i = 1
        job_num = len(self.wait_job)
        while (i < job_num):
            backfill_test = 0
            backfill_test = self.node_module.pre_avail(self.wait_job[i]['proc'],\
                    self.current_para['time'], self.current_para['time']+self.wait_job[i]['run'])
            if (backfill_test == 1):
                backfill_list.append(self.wait_job[i]['index'])
            self.node_module.reserve(self.wait_job[i]['proc'], self.wait_job[i]['index'], self.wait_job[i]['run'])
            i += 1  
        return backfill_list

    def backfill_RL(self):

        backfill_list = []  # List[Job Index] - Final selected backfillable job indices

        # Reserving First job.
        self.node_module.pre_reset(self.current_para['time'])       #会遍历当前时刻运行队列中的作业,记录运行作业完成时系统CPU资源的变化情况(该函数在后面回填作业时也可用到)
        self.node_module.reserve(self.wait_job[0]['proc'], self.wait_job[0]['index'], self.wait_job[0]['run'])      #为等待队列中第一个作业预留资源

        while True:
            backfill_candidates = []
            backfill_candidates_info = {}
            i = 1   #第一个作业已经在之前判断过是否可以为其预留资源
            job_num = len(self.wait_job)
            while i < job_num:
                backfill_test = self.node_module.pre_avail(self.wait_job[i]['proc'],
                                                           self.current_para['time'],
                                                           self.current_para['time'] + self.wait_job[i]['run'])
                if backfill_test == 1:
                    backfill_candidates.append(self.wait_job[i]['index'])
                    backfill_candidates_info[self.wait_job[i]['index']] = {
                        'pos': i,
                        'proc': self.wait_job[i]['proc'],
                        'index': self.wait_job[i]['index'],
                        'run': self.wait_job[i]['run'],
                    }
                i += 1

            if not backfill_candidates:
                break
            else:
                # ************* #
                # Reorder Queue Function [Cqsim_sim.reorder_queue()] - communicates with the Gym Environment and
                # pushes the selected job at the beginning.
                # ************* #
                wait_queue = self.current_para['reorder_queue_function'](backfill_candidates)
                selected_idx = wait_queue[0]
                backfill_list.append(selected_idx)
                self.node_module.reserve(backfill_candidates_info[selected_idx]['proc'],
                                         backfill_candidates_info[selected_idx]['index'],
                                         backfill_candidates_info[selected_idx]['run'])
                # 已将某个作业完成回填,将作业从等待队列中剔除
                self.wait_job = self.wait_job[:backfill_candidates_info[selected_idx]['pos']] + \
                                self.wait_job[backfill_candidates_info[selected_idx]['pos']+1:]

        return backfill_list

