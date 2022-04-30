import os
import time
from functools import partial
import threading
from timeit import default_timer as timer



class ParallelDependencyCache:
    directory = './cache'
    def init_dir(): os.makedirs(directory, exist_ok=True)
    
    def __init__(self, n_threads=1):
        self.n_threads = n_threads
        self.cfs = {}
        self.done = []
        self.levels = {}
        self.requiredby = {}
        self.refcounters = {}
        self.deps_ready = {}
        self.dont_delete = []
    
    def request(self, cf, dont_delete=True):        
        if cf.cached() and dont_delete:
            print('already cached:', cf.filename)
            return -1
        
        if dont_delete or cf.never_delete:
            self.dont_delete.append(cf.filename)
        
        if cf.filename in self.cfs.keys():
            return self.levels[cf.filename]
        
        self.cfs[cf.filename] = cf
        level = 0
        for dep in cf.deps:
            reclevel = self.request(dep, dont_delete=False) + 1
            level = max(reclevel, level)
            if dep.filename not in self.requiredby.keys():
                self.requiredby[dep.filename] = []
            self.requiredby[dep.filename].append(cf.filename)
        
        self.levels[cf.filename] = level
        self.deps_ready[cf.filename] = 0
        return level

    def fill_refcounters(self):
        for cf in self.requiredby:
            self.refcounters[cf] = len(self.requiredby[cf])
    
    def next_target(self):
        ready = [cf for cf in self.cfs.keys() \
            if self.deps_ready[cf] == len(self.cfs[cf].deps)]
        undone = [cf for cf in ready if cf not in self.done]
        def level(cf): return self.levels[cf]
        return sorted(undone, reverse=True, key=level)[:1]
    
    def decide_task_or_wait(self):
        task = ''
        wait = False
        
        taskarr = self.next_target()
        if len(taskarr) == 0:
            if len(self.done) != len(self.cfs):
                wait = True
        else:
            task = taskarr[0]
            self.done.append(task)
        
        return task,wait
    
    def fetch_all(self):
        self.fill_refcounters()
        plan_and_report = threading.Lock()
        def worker(workerid):
            while True:
                task = ''
                wait = False
                with plan_and_report:
                    task,wait = self.decide_task_or_wait()
                    if task: print('[%d]' % workerid, 'starting:', task)
                
                if not task and not wait: break
                elif not task and wait: time.sleep(0.1)
                else:
                    start = timer()
                    if not self.cfs[task].cached():
                        self.cfs[task].create_and_save()
                    with plan_and_report:
                        print('[%d]' % workerid, 'finished:', task, '(%.2f seconds)' % (timer() - start))
                        if task in self.requiredby:
                            for cf in self.requiredby[task]:
                                self.deps_ready[cf] += 1
                        for cf in self.cfs[task].deps:
                            self.refcounters[cf.filename] -= 1
                            if self.refcounters[cf.filename] == 0:
                                if not (cf.filename in self.dont_delete) and cf.cached():
                                    print('[%d]' % workerid, 'delete', cf.filename)
                                    self.cfs[cf.filename].delete()
        
        start = timer()
        threadpool = []
        for i in range(self.n_threads):
            t = threading.Thread(target=partial(worker, i+1))
            t.start()
            threadpool.append(t)
        for t in threadpool:
            t.join()
        print('TOTAL TIME: %.2f seconds' % (timer() - start))
        


class CachedFile:
    def __init__(self, classname, idparams, datatype, never_delete=False):
        self.deps = []
        self.filename = os.path.join(ParallelDependencyCache.directory, classname)
        if len(idparams) > 0:
            self.filename += '_' + '_'.join([str(e) for e in idparams])
        if datatype:
            self.filename += '.' + datatype
        self.never_delete = never_delete
    def recover(self): return None
    def create_and_save(self): return None
    def delete(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)
    def require(self, cf):
        self.deps.append(cf)
        return cf
    def cached(self):
        return os.path.isfile(self.filename)
    def get(self):
        if not self.cached():
            raise Exception('Error: %s has not been cached yet' % self.filename)
        return self.recover()
