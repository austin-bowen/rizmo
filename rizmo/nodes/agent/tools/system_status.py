import asyncio

import psutil
from easymesh.node.node import TopicSender

from rizmo.llm_utils import Tool


class GetSystemStatusTool(Tool):
    def __init__(self, say_topic: TopicSender, psutil_=psutil):
        self.say_topic = say_topic
        self.psutil = psutil_

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='get_system_status',
                description='Gets system CPU, memory, and disk usage, and CPU temperature in Celsius.',
            ),
        )

    async def call(self) -> dict:
        await self.say_topic.send('Checking...')
        return await asyncio.to_thread(self._get_system_status)

    def _get_system_status(self) -> dict:
        cpu_usage_percent = self.psutil.cpu_percent(interval=1.)
        memory_usage = self.psutil.virtual_memory()
        disk_usage = self.psutil.disk_usage('/')

        temps = psutil.sensors_temperatures()
        cpu_temp_c = (
            temps['thermal-fan-est'][0].current
            if 'thermal-fan-est' in temps else 'unknown'
        )

        return dict(
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_percent=memory_usage.percent,
            disk_usage_percent=disk_usage.percent,
            cpu_temp_c=cpu_temp_c,
        )
