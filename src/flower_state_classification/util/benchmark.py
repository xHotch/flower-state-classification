from functools import wraps
import time
import logging

def benchmark_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        total_start = time.perf_counter()
        result = func(*args, **kwargs)
        total_end = time.perf_counter()
        total_time = total_end - total_start

        logger = logging.getLogger(func.__name__)
        logger.info(f"Executing function {func.__name__} took: {total_time:.3f}s")
        wrapper.total_time = total_time
        return result
    return wrapper

def benchmark_fps(cooldown = 1):
    def decorator_fps(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            frame_start = time.perf_counter()
            result = func(*args, **kwargs)
            frame_end = time.perf_counter()
            frame_time = frame_end - frame_start
            wrapper.accum_time += frame_time
            wrapper.nr_frames += 1
            logger = logging.getLogger(func.__name__)

            if wrapper.accum_time > cooldown:
                avg_time = wrapper.accum_time / wrapper.nr_frames
                logger.debug(f"Average processing time: {avg_time:.3f}s ({1 / avg_time:.3f} fps)")
                wrapper.accum_time = 0
                wrapper.nr_frames = 0
            return result
        
        wrapper.nr_frames = 0
        wrapper.accum_time = 0
        return wrapper
    return decorator_fps