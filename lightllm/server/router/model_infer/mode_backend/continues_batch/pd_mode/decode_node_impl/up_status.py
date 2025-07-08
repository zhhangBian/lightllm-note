import time
import json
import asyncio
import threading
import websockets
import inspect

from typing import Dict
from dataclasses import asdict
from lightllm.server.pd_io_struct import UpKVStatus
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.pd_io_struct import PD_Master_Obj
import torch.multiprocessing as mp

logger = init_logger(__name__)


# 用于管理pd_master节点和decode节点之间的kv状态同步
class UpStatusManager:
    def __init__(self, args, task_in_queue: mp.Queue, task_out_queue: mp.Queue):
        self.args = args
        self.task_queue: mp.Queue[UpKVStatus] = task_in_queue
        self.task_out_queue = task_out_queue
        # 启动一个线程，用于处理kv状态同步
        self.daemon_thread = threading.Thread(target=self.thread_loop, daemon=True)
        self.daemon_thread.start()

    # 启动一个异步任务，用于处理kv状态同步
    def thread_loop(self):
        asyncio.run(self.task_loop())

    # 异步任务，用于处理kv状态同步
    async def task_loop(self):
        # 用于存储当前正在处理的kv状态同步任务
        self.id_to_handle_task: Dict[int, asyncio.Task] = {}
        # 用于存储当前正在处理的kv状态同步任务的队列
        self.id_to_handle_queue: Dict[int, asyncio.Queue] = {}

        # 创建一个异步任务，用于分发kv状态同步任务
        asyncio.create_task(self.dispatch_task_loop())

        while True:
            try:
                from lightllm.server.httpserver.pd_loop import _get_pd_master_objs

                # 获取所有的pd_master节点
                id_to_pd_master_obj = await _get_pd_master_objs(self.args)
                logger.info(f"get pd_master_objs {id_to_pd_master_obj}")

                # 如果获取到了pd_master节点，则需要处理当前正在处理的kv状态同步任务
                if id_to_pd_master_obj is not None:
                    # 遍历当前正在处理的kv状态同步任务
                    for node_id, pd_master_obj in self.id_to_handle_task.items():
                        # 如果当前正在处理的kv状态同步任务的pd_master节点不在获取到的pd_master节点中，则需要取消该任务
                        if node_id not in id_to_pd_master_obj:
                            self.id_to_handle_task[node_id].cancel()
                            self.id_to_handle_task.pop(node_id, None)
                            self.id_to_handle_queue.pop(node_id, None)
                            logger.info(f"up_kv_status_task {pd_master_obj} cancelled")

                    # 遍历获取到的pd_master节点
                    for node_id, pd_master_obj in id_to_pd_master_obj.items():
                        # 如果当前正在处理的kv状态同步任务的pd_master节点不在获取到的pd_master节点中，则需要创建相应的队列和任务
                        if node_id not in self.id_to_handle_task:
                            # 存储当前正在处理的kv状态同步任务
                            self.id_to_handle_queue[node_id] = asyncio.Queue()
                            # 处理当前正在处理的kv状态同步任务
                            # 创建一个异步任务，用于处理kv状态同步任务
                            self.id_to_handle_task[node_id] = asyncio.create_task(self.up_kv_status_task(pd_master_obj))

                await asyncio.sleep(30)

            except Exception as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    # 异步任务，用于分发kv状态同步任务
    async def dispatch_task_loop(self):
        while True:
            try:
                loop = asyncio.get_event_loop()
                upkv_status: UpKVStatus = await loop.run_in_executor(None, self.task_queue.get)
                if upkv_status.pd_master_node_id in self.id_to_handle_queue:
                    await self.id_to_handle_queue[upkv_status.pd_master_node_id].put(upkv_status)
                else:
                    logger.warning(f"upstatus {upkv_status} no connection to pd_master, drop it")
            except BaseException as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    # 异步任务，用于处理kv状态同步任务
    async def up_kv_status_task(self, pd_master_obj: PD_Master_Obj):
        while True:
            try:
                # 创建一个websocket连接，用于与pd_master节点进行通信
                uri = f"ws://{pd_master_obj.host_ip_port}/kv_move_status"
                async with websockets.connect(uri) as websocket:
                    import socket

                    sock = websocket.transport.get_extra_info("socket")
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    # 循环处理kv状态同步任务
                    while True:
                        try:
                            if pd_master_obj.node_id in self.id_to_handle_queue:
                                # 获取当前正在处理的kv状态同步任务的队列
                                task_queue = self.id_to_handle_queue[pd_master_obj.node_id]
                                # 获取当前正在处理的kv状态同步任务
                                upkv_status: UpKVStatus = await task_queue.get()
                                # 发送当前正在处理的kv状态同步任务
                                await websocket.send(json.dumps(asdict(upkv_status)))
                                logger.info(f"up status: {upkv_status}")
                            else:
                                await asyncio.sleep(3)
                        except BaseException as e:
                            logger.error(str(e))
                            raise e
            except asyncio.CancelledError:
                logger.info(f"up_kv_status_task {pd_master_obj} cancelled")
                return

            except Exception as e:
                logger.error(f"connetion to pd_master {pd_master_obj} has error: {str(e)}")
                logger.exception(str(e))
                await asyncio.sleep(10)
                logger.info("reconnection to pd_master")


def _init_env(args, task_in_queue: mp.Queue, task_out_queue: mp.Queue):
    graceful_registry(inspect.currentframe().f_code.co_name)
    # 创建一个UpStatusManager，用于管理pd_master节点和decode节点之间的kv状态同步
    up_kv_manager = UpStatusManager(args, task_in_queue, task_out_queue)
    logger.info(f"up kv manager {str(up_kv_manager)} start ok")
    while True:
        time.sleep(666)
    return


def start_up_kv_status_process(args, task_in_queue: mp.Queue, task_out_queue: mp.Queue):
    # 创建一个进程，用于管理pd_master节点和decode节点之间的kv状态同步
    proc = mp.Process(target=_init_env, args=(args, task_in_queue, task_out_queue))
    proc.start()
    assert proc.is_alive()
    logger.info("up_kv_status_process start")
    return proc
