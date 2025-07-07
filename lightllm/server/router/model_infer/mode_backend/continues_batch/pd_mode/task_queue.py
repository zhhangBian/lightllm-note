import threading
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# 由于PD分离的多进程特性，需要支持相应的锁进行调度
class TaskQueue:
    def __init__(self, get_func, fail_func):
        # 使用多线程的锁，避免多进程之间竞争
        self.lock = threading.Lock()
        self.datas = []
        self.get_func = get_func
        # 失败之后相应的回调函数
        self.fail_func = fail_func
        self.has_error = False

    def size(self):
        return len(self.datas)

    def put(self, obj, error_handle_func=None):
        if self.has_error:
            if error_handle_func is not None:
                error_handle_func(obj)
            raise Exception("has error")

        with self.lock:
            self.datas.append(obj)

    def put_list(self, objs):
        if self.has_error:
            raise Exception("has error")

        with self.lock:
            self.datas.extend(objs)

    def get_tasks(self, log_tag=None):
        with self.lock:
            ans = self.get_func(self.datas)
            self.datas = self.datas[len(ans) :]
        if len(self.datas) != 0:
            logger.info(f"queue {log_tag} left size: {len(self.datas)}")
        return ans

    def clear_tasks(self):
        with self.lock:
            if len(self.datas) != 0:
                for obj in self.datas:
                    self.fail_func(obj)
            self.datas = []
        return
