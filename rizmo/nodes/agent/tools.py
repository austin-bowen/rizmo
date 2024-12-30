import asyncio
import json
import subprocess
from dataclasses import asdict
from typing import Literal

import psutil
from openai.types.chat.chat_completion_message_tool_call import Function

from rizmo.nodes.agent.reminders import ReminderSystem
from rizmo.nodes.messages import MotorSystemCommand
from rizmo.weather import WeatherProvider


class ToolHandler:
    TOOLS = (
        # Template
        # dict(
        #     type='function',
        #     function=dict(
        #         name='name',
        #         description='Description of function.',
        #         parameters=dict(
        #             type='object',
        #             properties=dict(
        #                 arg=dict(
        #                     type='string',
        #                     description='Description of arg.',
        #                 ),
        #             ),
        #             required=['arg'],
        #             additionalProperties=False,
        #         ),
        #     ),
        # ),
        dict(
            type='function',
            description='Gets system CPU, memory, and disk usage, and CPU temperature in Celsius.',
            function=dict(
                name='get_system_status',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='get_weather',
                description=
                'Gets the weather for today, tomorrow, and the week, '
                'as well as the current moon phase.',
            ),
        ),
        dict(
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
        ),
        dict(
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
        ),
        dict(
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
        ),
    )

    def __init__(
            self,
            say_topic,
            motor_system_topic,
            weather_provider: WeatherProvider,
            reminder_system: 'ReminderSystem',
    ):
        self.say_topic = say_topic
        self.motor_system_topic = motor_system_topic
        self.weather_provider = weather_provider
        self.reminder_system = reminder_system

    async def handle(self, func_spec: Function) -> str:
        func = getattr(self, func_spec.name)
        kwargs = json.loads(func_spec.arguments)

        try:
            result = await func(**kwargs)
        except Exception as e:
            result = dict(error=repr(e))

        return json.dumps(result)

    async def get_system_status(self) -> dict:
        cpu_usage_percent = psutil.cpu_percent(interval=0.5)
        memory_usage = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')

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

    async def get_weather(self) -> dict:
        weather = await self.weather_provider.get_weather()
        return asdict(weather)

    async def motor_system(self, enabled: bool) -> None:
        await self.motor_system_topic.send(MotorSystemCommand(enabled=enabled))

    async def system_power(self, action: Literal['shutdown', 'reboot']) -> dict:
        if action == 'shutdown':
            return await self._shutdown()
        elif action == 'reboot':
            return await self._reboot()
        else:
            raise ValueError(f'Invalid action: {action}')

    async def _shutdown(self) -> dict:
        await self.say_topic.send('Shutting down.')
        await asyncio.sleep(3)

        return await self._run_cmd(
            'sudo', '--non-interactive', 'shutdown', '-h', 'now'
        )

    async def _reboot(self) -> dict:
        await self.say_topic.send('Be right back, rebooting.')
        await asyncio.sleep(3)

        return await self._run_cmd(
            'sudo', '--non-interactive', 'reboot'
        )

    async def reminders(self, action: str, reminder: str) -> list[str]:
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

    async def _run_cmd(self, *args) -> dict:
        result = subprocess.run(
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
