import asyncio


class DelayedCallback:
    """
    Used to schedule a callback to be called after a delay.
    Can be used to schedule the callback multiple times.
    """

    def __init__(self, delay: float, coro, *args, task_kwargs=None, **kwargs):
        """
        Args:
            delay: The delay in seconds before running the coroutine.
            coro: The coroutine function to schedule.
            *args: Positional arguments to pass to the coroutine.
            **kwargs: Keyword arguments to pass to the coroutine.
            task_kwargs: Optional keyword arguments to pass to `asyncio.create_task`.
        """

        self._delay = delay
        self._coro = coro
        self._args = args
        self._kwargs = kwargs
        self._task_kwargs = task_kwargs or {}
        self._task = None

    async def set(self, scheduled: bool) -> None:
        """Call `schedule()` if `scheduled` is `True`, otherwise call `cancel()`."""

        if scheduled:
            await self.schedule()
        else:
            await self.cancel()

    async def schedule(self):
        """
        Schedule the coroutine to run after the specified delay.

        If it has already been scheduled and is still pending/running,
        then this will do nothing.
        """

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run_after_delay(),
                **self._task_kwargs,
            )

    async def _run_after_delay(self):
        await asyncio.sleep(self._delay)
        await self._coro(*self._args, **self._kwargs)

    async def reschedule(self) -> None:
        await self.cancel()
        await self.schedule()

    async def cancel(self):
        """
        Cancel the scheduled coroutine if it is still pending.
        """

        task = self._task
        if task is None or task.done():
            return

        task.cancel()
        self._task = None

        try:
            await task
        except asyncio.CancelledError:
            pass
