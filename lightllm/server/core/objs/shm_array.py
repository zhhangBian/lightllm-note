import numpy as np
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# 用于管理共享内存中的数组
class ShmArray:
    def __init__(self, name, shape, dtype):
        self.shm = None
        self.arr = None
        self.name = name
        self.dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        self.dest_size = np.prod(shape) * self.dtype_byte_num
        self.shape = shape
        self.dtype = dtype

    # 创建共享内存空间
    def create_shm(self):
        try:
            shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.dest_size)
        except:
            shm = shared_memory.SharedMemory(name=self.name, create=False, size=self.dest_size)

        if shm.size != self.dest_size:
            logger.warning(f"size not same, unlink shm {self.name} and create again")
            shm.close()
            shm.unlink()
            try:
                shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.dest_size)
                logger.info(f"create shm {self.name}")
            except:
                shm = shared_memory.SharedMemory(name=self.name, create=False, size=self.dest_size)
                logger.info(f"link shm {self.name}")

        self.shm = shm  # SharedMemory 对象一定要被持有，否则会被释放
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    # 将共享内存空间链接到当前的数组中
    def link_shm(self):
        shm = shared_memory.SharedMemory(name=self.name, create=False, size=self.dest_size)
        assert shm.size == self.dest_size
        self.shm = shm
        self.arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        return

    # 关闭共享内存空间
    def close_shm(self):
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            self.arr = None
