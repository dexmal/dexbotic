import logging
import pickle
import grpc
import torch
import threading
import numpy as np
import cv2
from queue import Queue, Empty
from concurrent import futures
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks

try:
    from lerobot.async_inference.helpers import TimedAction
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class TimedAction:
        timestamp: float
        timestep: int
        action: torch.Tensor

from client import DexClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BridgeServer")

VLA_TARGET_URL = "http://10.0.0.1:7899"
ACTION_DIM = 6

class BridgeService(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, http_server_url, task_prompt):
        self.http_server_url = http_server_url
        self.task_prompt = task_prompt
        logger.info(f"Connecting to VLA Backend: {http_server_url}")
        self.dex_client = DexClient(base_url=http_server_url, use_delta=True)
        self.dex_client.set_init_action(np.zeros(ACTION_DIM))
        self.observation_queue = Queue(maxsize=1)
        self.environment_dt = 1/30.0
        self.shutdown_event = threading.Event()

    def Ready(self, request, context):
        logger.info("Robot Client Connected.")
        self.observation_queue = Queue(maxsize=1)
        self.dex_client.action_queue.clear()
        self.dex_client.set_init_action(np.zeros(ACTION_DIM))
        self.shutdown_event.clear()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):
        data = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, logger)
        obs = pickle.loads(data)
        if self.observation_queue.full():
            try:
                self.observation_queue.get_nowait()
            except Empty:
                pass
        self.observation_queue.put(obs)
        return services_pb2.Empty()
    
    def process_image_data(self, img_data):
        if img_data is None:
            return None
        if isinstance(img_data, torch.Tensor):
            img_data = img_data.permute(1, 2, 0).cpu().numpy()
        elif isinstance(img_data, np.ndarray):
            if img_data.ndim == 3 and img_data.shape[0] in [1, 3] and img_data.shape[2] > 3:
                img_data = np.transpose(img_data, (1, 2, 0))
        return img_data.astype(np.uint8)

    def GetActions(self, request, context):
        try:
            timed_obs = self.observation_queue.get(timeout=1.0)
            raw_obs = timed_obs.observation
            
            raw_img_front = raw_obs.get("front")
            raw_img_side = raw_obs.get("side")
            
            image_list = []
            img_front = self.process_image_data(raw_img_front)
            img_side = self.process_image_data(raw_img_side)
            
            if img_front is not None:
                cv2.imshow("Bridge - Front", cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR))
                image_list.append(img_front)
            if img_side is not None:
                cv2.imshow("Bridge - Side", cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR))
                image_list.append(img_side)
            
            if img_front is not None or img_side is not None:
                cv2.waitKey(1)
            
            client_obs = {'image': image_list}
            
            if len(self.dex_client.action_queue) == 0:
                if image_list:
                    self.dex_client.acquire_new_action(client_obs, self.task_prompt)
                else:
                    logger.warning(f"No images found. Keys: {list(raw_obs.keys())}")
            
            response_chunk = []
            chunk_start_ts = timed_obs.timestamp
            chunk_start_step = timed_obs.timestep
            
            idx = 0
            while len(self.dex_client.action_queue) > 0:
                act_np = self.dex_client.action_queue.popleft()
                if act_np.shape[0] != ACTION_DIM:
                    if act_np.shape[0] > ACTION_DIM:
                        act_np = act_np[:ACTION_DIM]
                    else:
                        pad = np.zeros(ACTION_DIM - act_np.shape[0])
                        act_np = np.concatenate([act_np, pad])
                
                t_action = TimedAction(
                    timestamp = chunk_start_ts + idx * self.environment_dt,
                    timestep = chunk_start_step + idx,
                    action = torch.from_numpy(act_np).float()
                )
                response_chunk.append(t_action)
                idx += 1
            
            if not response_chunk:
                return services_pb2.Empty()

            return services_pb2.Actions(data=pickle.dumps(response_chunk))

        except Empty:
            return services_pb2.Empty()
        except Exception as e:
            logger.error(f"Error: {e}")
            return services_pb2.Empty()

def serve(port, vla_url, prompt):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    bridge = BridgeService(vla_url, prompt)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(bridge, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Bridge Server started on [::]:{port}")
    logger.info(f"task: {prompt}")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vla_url", type=str, default=VLA_TARGET_URL)
    parser.add_argument("--prompt", type=str, default="Pick up the object")
    args = parser.parse_args()
    serve(args.port, args.vla_url, args.prompt)