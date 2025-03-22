import asyncio
import subprocess
from typing import Literal

from easymesh.node.node import TopicSender

from rizmo.llm_utils import Tool


class SystemPowerTool(Tool):
    def __init__(self, say_topic: TopicSender, subprocess_=subprocess):
        self.say_topic = say_topic
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
        await self.say_topic.send('Shutting down...')
        await asyncio.sleep(3)
        return await self._run_cmd('sudo', '--non-interactive', 'shutdown', '-h', 'now')

    async def _reboot(self) -> dict:
        await self.say_topic.send('Rebooting...')
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
