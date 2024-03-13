from datetime import datetime
import time
import re

import src_fc.CqSim.Node_struc as Class_Node_struc

__metaclass__ = type


class Node_struc_SWF(Class_Node_struc.Node_struc):        
        
    def node_allocate(self, proc_num, job_index, start, end):

        if self.is_available(proc_num) == 0:
            return 0

        i = 0
        for node in self.nodeStruc:
            if node['state'] < 0:
                node['state'] = job_index
                node['start'] = start
                node['end'] = end
                i += 1
            #self.debug.debug("  yyy: "+str(node['state'])+"   "+str(job_index),4)
            if (i>=proc_num):
                break

        self.idle -= proc_num
        self.avail = self.idle
        temp_job_info = {'job':job_index, 'end': end, 'node': proc_num}
        j = 0
        is_done = 0
        temp_num = len(self.job_list)
        while (j<temp_num):
            if (temp_job_info['end']<self.job_list[j]['end']):
                self.job_list.insert(j,temp_job_info)
                is_done = 1
                break
            j += 1

        if (is_done == 0):
            self.job_list.append(temp_job_info)
        '''
        self.debug.line(2,"...")
        for job in self.job_list:
            self.debug.debug(job['job'],2)
        self.debug.line(2,"...")
        '''
        self.debug.debug("  Allocate"+"["+str(job_index)+"]"+" Req:"+str(proc_num)+" Avail:"+str(self.avail)+" ",4)
        return 1

        
    def node_release(self, job_index, end):

        temp_node = 0
        j = 0
        temp_num = len(self.job_list)

        i = 0
        for node in self.nodeStruc:
            #self.debug.debug("  xxx: "+str(node['state'])+"   "+str(job_index),4)
            if node['state'] == job_index:
                node['state'] = -1
                node['start'] = -1
                node['end'] = -1
                i += 1
        if i <= 0:
            self.debug.debug("  Release Fail!",4)
            input("Problem !")
            return 0

        while (j<temp_num):
            if (job_index==self.job_list[j]['job']):
                temp_node = self.job_list[j]['node']
                break
            j += 1
        self.idle += temp_node
        self.avail = self.idle
        self.job_list.pop(j)
        self.debug.debug("  Release"+"["+str(job_index)+"]"+" Req:"+str(temp_node)+" Avail:"+str(self.avail)+" ",4)
        return 1

        
    def pre_avail(self, proc_num, start, end = None):
        #self.debug.debug("* "+self.myInfo+" -- pre_avail",6)
        #self.debug.debug("pre avail check: "+str(proc_num)+" (" +str(start)+";"+str(end)+")",6)
        if not end or end < start:
            end = start
             
        i = 0
        temp_job_num = len(self.predict_node)
        while (i < temp_job_num):
            if (self.predict_node[i]['time']>=start and self.predict_node[i]['time']<end):
                if (proc_num>self.predict_node[i]['avail']):    #proc_num表示想回填作业申请的核数,若想回填作业申请的核数大于正在运行任务结束时结点的资源情况，也就是说我回填的这个作业会将资源全部占完
                    return 0
            i += 1
        return 1
        
    def reserve(self, proc_num, job_index, time, start = None, index = -1 ):
        #proc_num表示作业申请核数 job_index表示作业下标 time表示作业预估运行时间 start
        #self.debug.debug("* "+self.myInfo+" -- reserve",5)
            
        temp_max = len(self.predict_node)
        if (start):
            if (self.pre_avail(proc_num,start,start+time)==0):
                return -1
        else:
            i = 0   #遍历正在运行作业队列情况
            j = 0
            if (index >= 0 and index < temp_max):
                i = index
            elif(index >= temp_max):
                return -1
            
            while (i<temp_max):
                if (proc_num<=self.predict_node[i]['avail']):  #predict_node列表第i个位置所代表的资源情况是否满足proc_num的资源要求
                    j = self.find_res_place(proc_num,i,time)
                    if (j == -1):
                        start = self.predict_node[i]['time']
                        break
                    else:
                        i = j + 1
                else:
                    i += 1

        end = start + time
        j = i
                
        is_done = 0
        start_index = j
        while (j < temp_max):
            if (self.predict_node[j]['time']<end):
                self.predict_node[j]['idle'] -= proc_num
                self.predict_node[j]['avail'] = self.predict_node[j]['idle']
                j += 1
            elif (self.predict_node[j]['time']==end):
                is_done = 1
                break
            else:
                self.predict_node.insert(j,{'time':end,\
                 'idle':self.predict_node[j-1]['idle'], 'avail':self.predict_node[j-1]['avail']})
                #self.debug.debug("xx   "+str(proc_num),4)
                self.predict_node[j]['idle'] += proc_num
                self.predict_node[j]['avail'] = self.predict_node[j]['idle']
                is_done = 1
                
                #self.debug.debug("xx   "+str(n)+"   "+str(k),4)
                break
            
        if (is_done != 1):
            self.predict_node.append({'time':end,'idle':self.tot,'avail':self.tot})
                
        self.predict_job.append({'job':job_index, 'start':start, 'end':end})
        '''
        i = 0
        self.debug.line(2,'.')
        temp_num = len(self.predict_node)
        self.debug.debug("<> "+str(job_index) +"   "+str(proc_num) +"   "+str(time) +"   ",2)
        while (i<temp_num):
            self.debug.debug("O "+str(self.predict_node[i]),2)
            i += 1
        self.debug.line(2,'.')
        ''' 
        return start_index
     
    def pre_delete(self, proc_num, job_index):
        #self.debug.debug("* "+self.myInfo+" -- pre_delete",5)
        return 1
        
    def pre_modify(self, proc_num, start, end, job_index):  
        #self.debug.debug("* "+self.myInfo+" -- pre_modify",5)  
        return 1
        
    def pre_get_last(self):
        #self.debug.debug("* "+self.myInfo+" -- pre_get_last",6)
        pre_info_last= {'start':-1, 'end':-1}
        for temp_job in self.predict_job:
            #self.debug.debug("xxx   "+str(temp_job),4)
            if (temp_job['start']>pre_info_last['start']):
                pre_info_last['start'] = temp_job['start']
            if (temp_job['end']>pre_info_last['end']):
                pre_info_last['end'] = temp_job['end']
        return pre_info_last
        
    def pre_reset(self, time):
        #self.debug.debug("* "+self.myInfo+" -- pre_reset",5)  
        self.predict_node = []
        self.predict_job = []
        self.predict_node.append({'time':time, 'idle':self.idle, 'avail':self.avail})
                            
                            
        temp_job_num = len(self.job_list)   #求出 正在运行的作业数量
        '''
        i = 0
        self.debug.line(2,'==')
        while (i<temp_job_num):
            self.debug.debug("[] "+str(self.job_list[i]),2)
            i += 1
        self.debug.line(2,'==')
        '''
        
        i = 0
        j = 0
        while i< temp_job_num:
            # predict_node列表里面代表每个运行作业完成时,系统所剩的计算资源情况
            if (self.predict_node[j]['time']!=self.job_list[i]['end'] or i == 0):
                self.predict_node.append({'time':self.job_list[i]['end'],\
                                    'idle':self.predict_node[j]['idle'], 'avail':self.predict_node[j]['avail']})
                j += 1
            self.predict_node[j]['idle'] += self.job_list[i]['node']
            self.predict_node[j]['avail'] = self.predict_node[j]['idle']
            i += 1
        ''' 
        i = 0
        self.debug.line(2,'..')
        temp_num = len(self.predict_node)
        while (i<temp_num):
            self.debug.debug("O "+str(self.predict_node[i]),2)
            i += 1
        self.debug.line(2,'..')
        '''
        return 1
        
    
    def find_res_place(self, proc_num, index, time):
        #self.debug.debug("* "+self.myInfo+" -- find_res_place",5)  
        if index>=len(self.predict_node):
            index = len(self.predict_node) - 1
             
        i = index
        end = self.predict_node[index]['time']+time
        temp_node_num = len(self.predict_node)
        
        while (i < temp_node_num):
            if (self.predict_node[i]['time']<end):
                if (proc_num>self.predict_node[i]['avail']):
                    #print "xxxxx   ",temp_node_num,proc_num,self.predict_node[i]
                    return i
            i += 1
        return -1