from threading import Thread

import numpy as np

from src_fc.ThreadMgr.Pause import Pause
import time
__metaclass__ = type


class Cqsim_sim(Pause, Thread):

    def __init__(self, module, debug=None):
        Thread.__init__(self)
        Pause.__init__(self)

        self.myInfo = "Cqsim Sim"
        self.alg_str = module['info'].alg_str
        self.module = module
        self.debug = debug
        
        self.debug.line(4," ")
        self.debug.line(4,"#")
        self.debug.debug("# "+self.myInfo,1)
        self.debug.line(4,"#")
        
        self.event_seq = []
        self.current_event = None
        self.reserve_job_id = -1
        #obsolete
        self.job_num = len(self.module['job'].job_info())
        self.currentTime = 0
        #obsolete
        self.read_job_buf_size = 100
        self.read_job_pointer = 0 # next position in job list
        self.previous_read_job_time = -1 # lastest read job submit time
        
        self.debug.line(4)
        for module_name in self.module:
            temp_name = self.module[module_name].myInfo
            self.debug.debug(temp_name+" ................... Load",4)
            self.debug.line(4)

        # ************ #
        # Shared Variables to communicate with GymEnvironment.
        # ************ #
        self.simulator_wait_que_indices = []
        self.simulator_cpu_count = 0
        self.is_simulation_complete = False

        self.jobSelectList = []
        self.jobAllocList = []

        self.minAutoAllocNodes = 90

    def run(self) -> None:
        """
        Invoke thread which runs the CqSim.
        :return:
        """
        self.cqsim_sim()

    def reset(self, module=None, debug=None):

        if module:
            self.module = module
        
        if debug:
            self.debug = debug

        self.event_seq = []
        self.current_event = None
        self.reserve_job_id = -1
        # obsolete
        self.job_num = len(self.module['job'].job_info())
        self.currentTime = 0
        # obsolete
        self.read_job_buf_size = 100
        self.read_job_pointer = 0
        self.previous_read_job_time = -1

    def cqsim_sim(self):

        self.import_submit_events()
        self.scan_event()

        self.print_result()

        self.is_simulation_complete = True
        self.release_all()

        print('作业选择算法平均时间为:',np.mean(self.jobSelectList),'作业资源分配算法平均时间为:',np.mean(self.jobAllocList))
        self.debug.debug("------ Simulating Done!", 2)
        self.debug.debug(lvl=1)
        return

    def import_submit_events(self):
        # fread jobs to job list and buffer to event_list dynamically
        if self.read_job_pointer < 0:
            return -1
        temp_return = self.module['job'].dyn_import_job_file()  #temp_return只能返回-1/0,返回-1代表需要读取的文件已关闭。返回0代表读取文件的一行信息。
        i = self.read_job_pointer
        #while (i < len(self.module['job'].job_info())):
        while (i < self.module['job'].job_info_len()):
            self.insert_event(1,self.module['job'].job_info(i)['submit'],2,[1,i])   #向event_seq这个event列表中添加event对象,最后一个参数表示para;para的第一个元素表示作业状态,1表示submit状态.i表示所读取作业的index
            self.previous_read_job_time = self.module['job'].job_info(i)['submit']
            self.debug.debug("  "+"Insert job["+"2"+"] "+str(self.module['job'].job_info(i)['submit']),4)
            i += 1

        if temp_return == None or temp_return < 0 :
            self.read_job_pointer = -1
            return -1
        else:
            self.read_job_pointer = i   #可能在while循环中将i的值改变,需要重新给read_job_pointer变量赋值;在此函数开始代码处,会把self.read_job_pointer赋值给i
            return 0
    
    def insert_event(self, type, time, priority, para = None):
        #self.debug.debug("# "+self.myInfo+" -- insert_event",5) 
        temp_index = -1
        new_event = {"type":type, "time":time, "prio":priority, "para":para}    #para代表event是何种类型;若为submit,则0号元素为1;若为finish,则0号元素为2;
        flag = False # 用于判断是否有相同时刻且都为Finsh的Event存在.
        if (type == 1):
            i = 0   #遍历 event_seq找一个合适的位置,新加入的event需要在优先级函数与时间维度上一起判断(若时间相同,则比较优先级,优先级低的在前面;若时间不同,时间越小的排在前面)
            while (i<len(self.event_seq)):
                if (self.event_seq[i]['time']==time):
                    # if (self.event_seq[i]['prio']>priority):
                    temp_index = i
                    # 首先满足Event时间在同一时刻,其次满足状态为Finish
                    if self.event_seq[i]['para'][0] == 2:
                        if len(self.event_seq[i]['para']) >= 2 and para[0] == 2:
                            self.event_seq[i]['para'].append(para[1]);
                            flag = True
                    break
                elif (self.event_seq[i]['time']>time):
                    temp_index = i
                    break 
                i += 1
        if not flag:
            if (temp_index>=len(self.event_seq) or temp_index == -1):
                self.event_seq.append(new_event)    #将新的event加入到列表的最后
            else:
                self.event_seq.insert(temp_index,new_event)     #将新的event加入到event_seq中第一个时间>time的位置,即新加入的event的发生时间要小于原有event_seq中的元素的发生时间

    def scan_event(self):

        self.debug.line(2, " ")
        self.debug.line(2, "=")
        self.debug.line(2, "=")
        self.current_event = None
        while (len(self.event_seq) > 0 or self.read_job_pointer >= 0) or self.module['job'].job_wait_list:
            if (len(self.event_seq) > 0 or self.read_job_pointer >= 0):
                while (len(self.event_seq) > 0 or self.read_job_pointer >= 0):
                    if len(self.event_seq) > 0:
                        temp_current_event = self.event_seq[0]
                        temp_currentTime = temp_current_event['time']
                    else:
                        temp_current_event = None
                        temp_currentTime = -1
                    if temp_current_event:
                        temp_current_event['state'] = 0


                    if (len(self.event_seq) == 0 or temp_currentTime >= self.previous_read_job_time) and self.read_job_pointer >= 0:        #若read_job_pointer小于0,即只有-1,-1表示读取不到文件中的数据,即若文件中有数据则读取;之前括号里有两条件,event_seq中有event且event队列中第一个元素所对应的时间<上一个读取作业的时间
                        self.import_submit_events()
                        continue

                    self.current_event = temp_current_event
                    self.currentTime = temp_currentTime

                    if self.current_event['type'] == 1:

                        self.debug.line(2, " ")
                        self.debug.line(2, ">>>")
                        self.debug.line(2, "--")
                        self.debug.debug("  Time: "+str(self.currentTime), 2)
                        self.debug.debug("   "+str(self.current_event), 2)
                        self.debug.line(2, "--")
                        self.debug.debug("  Wait: "+str(self.module['job'].wait_list()),2)
                        self.debug.debug("  Run : "+str(self.module['job'].run_list()),2)
                        self.debug.line(2, "--")
                        self.debug.debug("  Tot:"+str(self.module['node'].get_tot())+" Idle:"+str(self.module['node'].get_idle())+" Avail:"+str(self.module['node'].get_avail())+" ",2)
                        self.debug.line(2, "--")

                        self.event_job(self.current_event['para'])  #Event的para参数指定此Event的类型,根据Event的类型判断是submit/finish
                        #       将操作过的Event标记为1
                        self.current_event['state'] = 1
                    self.sys_collect()
                    # 删除当前时间点之前的所有event
                    index_list = []
                    for index,eventInList in enumerate(self.event_seq):
                        if eventInList['time'] < self.currentTime:
                            index_list.append(index)
                        elif eventInList['time'] == self.currentTime:
                            if 'state' in eventInList and eventInList['state'] == 1:
                                index_list.append(index)
                    for index in sorted(index_list,reverse=True):
                        del self.event_seq[index]
            elif self.module['job'].job_wait_list:
                if self.alg_str == 'SJF' or self.alg_str == 'FCFS':
                    self.start_scan_forRigid()
                else:
                    self.start_scan()  # 在等待队列的start_window中选择作业开始执行/回填作业

        self.debug.line(2,"=")
        self.debug.line(2,"=")
        self.debug.line(2," ")
        return
    
    def event_job(self, para_in = None):
        #  有作业提交/删除都需要重新start_scan
        if (self.current_event['para'][0] == 1):
            self.submit(self.current_event['para'][1])
        elif (self.current_event['para'][0] == 2):
            if len(self.current_event['para']) - 1 <= 1:
                self.finish(self.current_event['para'][1])
            else:
                index_list = self.current_event['para'][1:]
                for index in index_list:
                    self.finish(index)
        # Obsolete
        # self.score_calculate()
        # 若是强化学习调度算法根据start_scan()处理,若是rigid调度算法则根据start_scanForRigid()
        if self.alg_str == 'SJF' or self.alg_str == 'FCFS':
            self.start_scan_forRigid()
        else:
            self.start_scan()   #在等待队列的start_window中选择作业开始执行/回填作业
    
    def submit(self, job_index):
        #self.debug.debug("# "+self.myInfo+" -- submit",5) 
        # self.debug.debug("[Submit]  "+str(job_index),3)
        self.debug.debug("[Submit]  " + str(job_index)+" "+str(self.module['job'].job_info(job_index)), 3)
        self.module['job'].job_submit(job_index)
        return
    
    def finish(self, job_index):
        #self.debug.debug("# "+self.myInfo+" -- finish",5) 
        self.debug.debug("[Finish]  "+str(job_index),3)
        self.module['node'].node_release(job_index,self.currentTime)
        self.module['job'].job_finish(job_index)
        self.module['output'].print_result(self.module['job'], job_index)
        self.module['job'].remove_job_from_dict(job_index)
        return
    
    def start_job(self, job_index):
        # self.debug.debug("# "+self.myInfo+" -- start",5)
        self.debug.debug("[Start]  "+str(job_index), 3)
        self.module['node'].node_allocate(self.module['job'].job_info(job_index)['autoAllocNode'], job_index,
                                          self.currentTime, self.currentTime +
                                          self.module['job'].job_info(job_index)['estRunTime'])
        self.module['job'].job_start(job_index, self.currentTime)
        self.debug.debug("[Finish Debug Time]  " + "Curr Time:"+str(self.currentTime)+" runTime:"+str(self.module['job'].job_info(job_index)['run'])+" Total:"+str(self.currentTime+self.module['job'].job_info(job_index)['run']), 3)
        self.insert_event(1, self.currentTime+self.module['job'].job_info(job_index)['run'], 1, [2, job_index])
        return

    def getAllocNum(self, wait_que):
        """
        This(and only this) function manages thread synchronization and communication with the GymEnvironment.

        :param wait_que: [List[int]] : CqSim WaitQue at current Time.
        :return: [List[int]] : Updated wait_que, with the selected job at the beginning.

        模拟调度系统将等待队列准备好了，要将等待队列传入到gym环境中,让gym中的智能体根据等待队列中作业情况,综合考虑为作业分配合适资源
        """
        self.simulator_wait_que_indices = wait_que
        self.pause_consumer()
        return

    #   需要在等待队列中选择任务执行
    def start_scan(self):

        start_max = self.module['win'].start_num()
        temp_wait = self.module['job'].wait_list()   #从调度模拟环境 job模块中取等待队列
        win_count = start_max

        while temp_wait:        #等待队列的列表不为空开始循环
            if win_count >= start_max:
                win_count = 0
                temp_wait = self.start_window(temp_wait)

            # ************ #
            # Communicate with GymEnvironment. 调度系统已经把等待队列准备好,放入到共享变量simulator_wait_que_indices中,现在需要将线程调转到gym调度上
            # ************ #

            # 当前有空闲资源时,agent需要继续完成资源分配

            # 需要将作业调度与资源分配按照不同的算法执行,资源分配使用DRL,作业选择使用deadLine感知.
            print("Wait Queue at StartScan - ", temp_wait)
            start_time_jobSelect = time.time()
            # 根据deadLine QoS需求选择作业
            minDeadLineQoSIndex = -1
            minQoSAware = float('inf')
            for index,num in enumerate(temp_wait):
                tempJobInfo = self.module['job'].job_info(num)
                qosRemain = tempJobInfo['qosRunTime'] - (self.currentTime - tempJobInfo['submit'])
                if qosRemain < minQoSAware:
                    minQoSAware = qosRemain
                    minDeadLineQoSIndex = index
            temp_num = temp_wait[0]
            temp_wait[0] = temp_wait[minDeadLineQoSIndex]
            temp_wait[minDeadLineQoSIndex] = temp_num

            end_time_jobSelect = time.time()
            jobSelect = end_time_jobSelect - start_time_jobSelect

            if self.module['node'].idle == 0 or self.module['node'].idle <= self.minAutoAllocNodes:
                break

            start_time_alloc = time.time()
            allocNum = 0
            flag = False
            while allocNum < 5:
                # 使用RL_Agent选择资源分配方案
                self.selectJobInfo = self.module['job'].job_info(temp_wait[0])
                self.getAllocNum(temp_wait)
                # print("智能体线程所选择资源分配的结果为:", self.selectJobInfo['autoAllocNode'])
                temp_job_id = temp_wait[0]
                temp_job = self.module['job'].job_info(temp_job_id)
                if temp_job['autoAllocNode'] < 0:
                    flag = True
                    break
                if self.module['node'].is_available(temp_job['autoAllocNode']):
                    # print(f'temp_job_id: {temp_job_id}')  若系统中计算资源数量满足该作业所申请的核数
                    if self.reserve_job_id == temp_job_id:
                        self.reserve_job_id = -1

                    self.start_job(temp_job_id)
                    self.debug.debug("[AutoAllocResource]  " + str(temp_job['autoAllocNode']), 3)
                    temp_wait.pop(0)
                    # 本层的break 用于 中止分配资源循环
                    break
                else:
                    allocNum+=1
            end_time_alloc = time.time()
            jobAlloc = end_time_alloc - start_time_alloc
            self.jobSelectList.append(jobSelect)
            self.jobAllocList.append(jobAlloc)
            # print('作业选择时间:',jobSelect,'作业资源分配时间:',jobAlloc)
            if flag:
                break
            # 外层的break 从外层的while循环中出来,主要用于扫描等待队列
            # break

            win_count += 1
        return
    def start_scan_forRigid(self):

        start_max = self.module['win'].start_num()
        temp_wait = self.module['job'].wait_list()   #从调度模拟环境 job模块中取等待队列
        win_count = start_max

        while temp_wait:        #等待队列的列表不为空开始循环
            if win_count >= start_max:
                win_count = 0
                temp_wait = self.start_window(temp_wait)

            # ************ #
            # Communicate with GymEnvironment. 调度系统已经把等待队列准备好,放入到共享变量simulator_wait_que_indices中,现在需要将线程调转到gym调度上
            # ************ #
            print("Wait Queue at StartScan - ", temp_wait)
            if temp_wait[0] != self.reserve_job_id:
                temp_wait = self.reorder_queue(temp_wait)

            temp_job_id = temp_wait[0]      #会将RL智能体在Action下取到的作业放到队列的最前面
            temp_job = self.module['job'].job_info(temp_job_id)
            if self.module['node'].is_available(temp_job['autoAllocNode']):
                # print(f'temp_job_id: {temp_job_id}')  若系统中计算资源数量满足该作业所申请的核数
                if self.reserve_job_id == temp_job_id:
                    self.reserve_job_id = -1

                self.start_job(temp_job_id)
                temp_wait.pop(0)
            else:
                # 若资源数量不满足
                temp_wait = self.module['job'].wait_list()  # 将等待队列保存到temp_wait中
                self.reserve_job_id = temp_wait[0]
                self.backfill(temp_wait)
                break

            win_count += 1
        return

    def reorder_queue(self, wait_que):
        """
        This(and only this) function manages thread synchronization and communication with the GymEnvironment.

        :param wait_que: [List[int]] : CqSim WaitQue at current Time.
        :return: [List[int]] : Updated wait_que, with the selected job at the beginning.

        模拟调度系统将等待队列准备好了，要将等待队列传入到gym环境中，让gym从等待队列中选择job执行
        """
        self.simulator_wait_que_indices = wait_que
        self.pause_consumer()
        return self.simulator_wait_que_indices

    def start_window(self, temp_wait_B):    #在等待队列中截window_size个作业

        win_size = self.module['win'].window_size()
        #   传进来的tmp_wait_B若里面的元素数量大于window_module所规定的大小,需要截断
        if (len(temp_wait_B)>win_size):
            temp_wait_A = temp_wait_B[0:win_size]
            temp_wait_B = temp_wait_B[win_size:]
        else:
            temp_wait_A = temp_wait_B
            temp_wait_B = []

        temp_wait_info = []
        max_num = len(temp_wait_A)
        i = 0
        while i < max_num:  #遍历tmp_wait_A中的所有元素并加入到tmp_wait_info中
            temp_job = self.module['job'].job_info(temp_wait_A[i])
            temp_wait_info.append({"index": temp_wait_A[i], "proc": temp_job['autoAllocNode'],
                                   "node": temp_job['autoAllocNode'], "run": temp_job['run'],
                                   "score": temp_job['score']})
            i += 1
            
        temp_wait_A = self.module['win'].start_window(temp_wait_info,{"time":self.currentTime})
        temp_wait_B[0:0] = temp_wait_A
        return temp_wait_B
    
    def backfill(self, temp_wait):
        temp_wait_info = []
        max_num = len(temp_wait)
        i = 0
        while i < max_num:
            temp_job = self.module['job'].job_info(temp_wait[i])
            temp_wait_info.append({"index": temp_wait[i], "proc": temp_job['autoAllocNode'],
                                   "node": temp_job['autoAllocNode'], "run": temp_job['run'], "score": temp_job['score']})
            i += 1

        # ************ #
        # reorder_queue function passed as an argument, to be invoked while selecting back-fill jobs.(reorder_queue函数主要可以将线程交换到RL智能选择作业的主线程上,故叫做重新排布作业顺序函数)
        # ************ #
        backfill_list = self.module['backfill'].backfill(temp_wait_info, {'time': self.currentTime,
                                                                          'reorder_queue_function': self.getAllocNum})

        if not backfill_list:
            return 0
        
        for job in backfill_list:
            print('backfill job.')
            self.start_job(job)
        return 1
    
    def sys_collect(self):

        temp_inter = 0
        if (len(self.event_seq) > 1):
            temp_inter = self.event_seq[1]['time'] - self.currentTime
        temp_size = 0
        
        event_code=None
        if (self.event_seq[0]['type'] == 1):
            if (self.event_seq[0]['para'][0] == 1):   
                event_code='S'
            elif(self.event_seq[0]['para'][0] == 2):   
                event_code='E'
        elif (self.event_seq[0]['type'] == 2):
            event_code='Q'
        temp_info = self.module['info'].info_collect(time=self.currentTime, event=event_code,
                                                     uti=(self.module['node'].get_tot() -
                                                          self.module['node'].get_idle()) *
                                                         1.0/self.module['node'].get_tot(),
                                                     waitNum=len(self.module['job'].wait_list()),
                                                     waitSize=self.module['job'].wait_size(), inter=temp_inter)
        self.print_sys_info(temp_info)
        return

    def print_sys_info(self, sys_info):
        self.module['output'].print_sys_info(sys_info)
    
    def print_result(self):
        self.module['output'].print_sys_info()
        self.debug.debug(lvl=1)
        self.module['output'].print_result(self.module['job'])
