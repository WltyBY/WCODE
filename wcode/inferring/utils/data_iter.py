import os
import torch
import queue
import multiprocessing

from time import sleep
from torch.multiprocessing import Event, Queue, Manager

from wcode.preprocessing.preprocessor import Preprocessor


def preprocess_fromfiles_save_to_queue(
    images_dict: dict,
    preprocess_config: str,
    predictions_save_folder: str,
    dataset_name: str,
    target_queue: Queue,
    done_event: Event,
    abort_event: Event,
    verbose: bool,
):
    try:
        preprocessor = Preprocessor(dataset_name=dataset_name, verbose=verbose)
        for key in images_dict.keys():
            data, _, data_properites = preprocessor.run_case(images_dict[key], None, preprocess_config)

            data = torch.from_numpy(data).contiguous().float()

            item = {
                "data": data,
                "data_properites": data_properites,
                "output_file": os.path.join(predictions_save_folder, key),
            }
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(
    images_dict: dict,
    preprocess_config: str,
    predictions_save_folder: str,
    dataset_name: str,
    pin_memory: bool = False,
    num_processes: int = 8,
    verbose: bool = False,
):
    context = multiprocessing.get_context("spawn")
    manager = Manager()
    num_processes = min(len(images_dict), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        single_work_img_dict = dict(list(images_dict.items())[i::num_processes])
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(
            target=preprocess_fromfiles_save_to_queue,
            args=(
                single_work_img_dict,
                preprocess_config,
                predictions_save_folder,
                dataset_name,
                queue,
                event,
                abort_event,
                verbose,
            ),
            daemon=True,
        )
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (
        not target_queues[worker_ctr].empty()
    ):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = (
                all(
                    [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]
                )
                and not abort_event.is_set()
            )
            if not all_ok:
                raise RuntimeError(
                    "Background workers died. Look for the error message further up! If there is "
                    "none then your RAM was full and the worker was killed by the OS. Use fewer "
                    "workers or get more RAM in that case!"
                )
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
