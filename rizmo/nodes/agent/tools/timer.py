import asyncio
from asyncio import Task
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from easymesh.utils import require

from rizmo.llm_utils import Tool


class TimerTool(Tool):
    def __init__(self, timer_complete_callback):
        self.timer_complete_callback = timer_complete_callback

        self.timers: dict['Timer', Task] = {}

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='timers',
                description='Manages timers.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        action=dict(
                            type='string',
                            description='The action to perform: '
                                        '`start` starts a new timer; '
                                        '`stop` stops a specific timer; '
                                        '`stop_all` stops all timers; '
                                        '`list` returns a list of all active timers.',
                            enum=['start', 'stop', 'stop_all', 'list'],
                        ),
                        duration=dict(
                            type='object',
                            description='Duration of the timer.',
                            properties=dict(
                                hours=dict(
                                    type='integer',
                                    minimum=0,
                                ),
                                minutes=dict(
                                    type='integer',
                                    minimum=0,
                                ),
                                seconds=dict(
                                    type='integer',
                                    minimum=0,
                                ),
                            ),
                        ),
                        name=dict(
                            type='string',
                            description='Optional name for the timer.',
                        ),
                    ),
                    required=['action'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: str, duration: dict = None, name: str = None):
        if action == 'start':
            timer = Timer.from_duration_dict(duration, name)
            require(timer.duration.total_seconds() > 0)
            await self._start_timer(timer)
            return f'Started {timer}'

        elif action == 'stop':
            timer = Timer.from_duration_dict(duration, name)
            await self._stop_timer(timer)
            return f'Stopped {timer}'

        elif action == 'stop_all':
            await self._stop_all_timers()
            return 'Stopped all timers'

        elif action == 'list':
            return await self._list_timers()

        else:
            raise ValueError(f'Invalid action: {action}')

    async def _start_timer(self, timer: 'Timer') -> None:
        self.timers[timer] = asyncio.create_task(self._run_timer(timer))

    async def _run_timer(self, timer: 'Timer') -> None:
        await asyncio.sleep(timer.duration.total_seconds())
        del self.timers[timer]
        await self.timer_complete_callback(timer)

    async def _stop_timer(self, timer: 'Timer') -> None:
        task = self.timers.pop(timer, None)
        if task:
            task.cancel()

    async def _stop_all_timers(self) -> None:
        tasks = list(self.timers.values())
        self.timers.clear()
        for task in tasks:
            task.cancel()

    async def _list_timers(self) -> list[str]:
        timers = []

        for timer in self.timers.keys():
            remaining = timer.remaining
            remaining = (
                timedelta_to_hms_text(remaining, plural=True)
                if remaining else 'no time'
            )

            timers.append(f'{timer}: {remaining} remaining')

        return timers


@dataclass(frozen=True)
class Timer:
    duration: timedelta
    name: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def end_time(self) -> datetime:
        return self.start_time + self.duration

    @property
    def remaining(self) -> Optional[timedelta]:
        remaining = self.end_time - datetime.now()
        return remaining if remaining.total_seconds() > 0 else None

    @classmethod
    def from_duration_dict(cls, duration: dict, name: Optional[str]) -> 'Timer':
        duration = timedelta(
            hours=duration.get('hours', 0),
            minutes=duration.get('minutes', 0),
            seconds=duration.get('seconds', 0),
        )

        return cls(duration, name)

    def __str__(self) -> str:
        duration = timedelta_to_hms_text(self.duration, plural=False)

        name = f'called "{self.name}"' if self.name else '(no name)'

        return f'{duration} timer {name}'


def timedelta_to_hms_text(td: timedelta, *, plural: bool) -> str:
    h, m, s = timedelta_to_hms_components(td)

    duration = []
    if h > 0:
        duration.append(f'{h} hour')
    if m > 0:
        duration.append(f'{m} minute')
    if s > 0 or (h == 0 and m == 0):
        duration.append(f'{s} second')

    if plural:
        duration = [d + 's' for d in duration]

    return ' '.join(duration)


def timedelta_to_hms_components(td: timedelta) -> tuple[int, int, int]:
    """Convert a timedelta to (hours, minutes, seconds)."""
    total_seconds = round(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds
