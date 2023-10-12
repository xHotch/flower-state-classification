# Code adapted from https://realpython.com/python-timer/

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, List[float]]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = None
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.text = f"{self.name}: {self.text}"
            self.timers.setdefault(self.name, [])

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name].append(elapsed_time)

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()

    @classmethod
    def get_total_time(cls, timer_name) -> float:
        """Get the total time of a timer"""
        if timer_name not in cls.timers:
            raise TimerError(f"Timer {timer_name} does not exist")
        return sum(cls.timers[timer_name])

    @classmethod
    def get_number_of_executions(cls, timer_name) -> int:
        """Get the number of executions of a timer"""
        if timer_name not in cls.timers:
            raise TimerError(f"Timer {timer_name} does not exist")
        return len(cls.timers[timer_name])

    @classmethod
    def get_average_time(cls, timer_name) -> float:
        """Get the average time of a timer"""
        if timer_name not in cls.timers:
            raise TimerError(f"Timer {timer_name} does not exist")
        return cls.get_total_time(timer_name) / cls.get_number_of_executions(timer_name)

    @classmethod
    def has_timer(cls, timer_name) -> bool:
        """Check if a timer exists"""
        return timer_name in cls.timers

    @classmethod
    def print_summary(cls, logging_function: Callable = print) -> None:
        """Print a summary of all timers"""
        for timer_name, values in cls.timers.items():
            if values:
                logging_function(
                    f"Total/Average time for {timer_name}: {cls.get_total_time(timer_name):.3f}s/{cls.get_average_time(timer_name):.3f}s . ({cls.get_number_of_executions(timer_name)} executions)"
                )
