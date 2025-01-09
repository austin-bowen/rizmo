import asyncio
import subprocess
from dataclasses import asdict
from typing import Literal

import psutil
from easymesh.node.node import MeshNode

from rizmo.config import config
from rizmo.llm_utils import Tool, ToolHandler
from rizmo.nodes.agent.reminders import ReminderSystem
from rizmo.nodes.messages import MotorSystemCommand
from rizmo.nodes.topics import Topic
from rizmo.weather import WeatherProvider


def get_tool_handler(node: MeshNode) -> ToolHandler:
    weather_provider = WeatherProvider.build(config.weather_location)

    reminder_system = ReminderSystem(config.reminders_file_path)

    return ToolHandler([
        GetSystemStatusTool(),
        GetWeatherTool(weather_provider),
        MotorSystemTool(node.get_topic_sender(Topic.MOTOR_SYSTEM)),
        SystemPowerTool(),
        RemindersTool(reminder_system),
    ])


class GetSystemStatusTool(Tool):
    def __init__(self, psutil_=psutil):
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
        return await asyncio.to_thread(self._get_system_status)

    def _get_system_status(self) -> dict:
        cpu_usage_percent = self.psutil.cpu_percent(interval=0.5)
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


class GetWeatherTool(Tool):
    def __init__(self, weather_provider: WeatherProvider):
        self.weather_provider = weather_provider

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='get_weather',
                description=
                'Gets the weather for today, tomorrow, and the week, '
                'as well as the current moon phase.',
            ),
        )

    async def call(self) -> dict:
        weather = await self.weather_provider.get_weather()
        return asdict(weather)


class MotorSystemTool(Tool):
    def __init__(self, motor_system_topic):
        self.motor_system_topic = motor_system_topic

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='motor_system',
                description='Controls the motor system.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        enabled=dict(
                            type='boolean',
                            description='Whether to enable or disable the motor system.',
                        ),
                    ),
                    required=['enabled'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, enabled: bool) -> None:
        await self.motor_system_topic.send(MotorSystemCommand(enabled=enabled))


class SystemPowerTool(Tool):
    def __init__(self, subprocess_=subprocess):
        self.subprocess = subprocess_

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='system_power',
                description='Shutdown or reboot the system',
                parameters=dict(
                    type='object',
                    properties=dict(
                        action=dict(
                            type='string',
                            description='The action to perform.',
                            enum=['shutdown', 'reboot'],
                        ),
                    ),
                    required=['action'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: Literal['shutdown', 'reboot']) -> dict:
        if action == 'shutdown':
            return await self._shutdown()
        elif action == 'reboot':
            return await self._reboot()
        else:
            raise ValueError(f'Invalid action: {action}')

    async def _shutdown(self) -> dict:
        await asyncio.sleep(3)
        return await self._run_cmd('sudo', '--non-interactive', 'shutdown', '-h', 'now')

    async def _reboot(self) -> dict:
        await asyncio.sleep(3)
        return await self._run_cmd('sudo', '--non-interactive', 'reboot')

    async def _run_cmd(self, *args: str) -> dict:
        return await asyncio.to_thread(self._run_cmd_sync, args)

    def _run_cmd_sync(self, args: tuple[str, ...]) -> dict:
        result = self.subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return dict(
            command=' '.join(args),
            exit_code=result.returncode,
            stdout=result.stdout.decode(),
            stderr=result.stderr.decode(),
        )


class RemindersTool(Tool):
    def __init__(self, reminder_system: ReminderSystem):
        self.reminder_system = reminder_system

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='reminders',
                description='Manages the reminders system and returns the list of reminders after the action is performed.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        action=dict(
                            type='string',
                            description='The action to perform.',
                            enum=['list', 'add', 'remove', 'clear'],
                        ),
                        reminder=dict(
                            type='string',
                            description='The reminder to add/remove. Ignored if action is "list" or "clear"',
                        ),
                    ),
                    required=['action', 'reminder'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: str, reminder: str) -> list[str]:
        if action == 'list':
            pass
        elif action == 'add':
            self.reminder_system.add(reminder)
        elif action == 'remove':
            self.reminder_system.remove(reminder)
        elif action == 'clear':
            self.reminder_system.clear()
        else:
            raise ValueError(f'Invalid action: {action}')

        return self.reminder_system.list()
