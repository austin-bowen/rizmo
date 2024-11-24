"""
Limits:
0: 400 - 2600
1: 512 - 2488
2: 512 - 2208
"""

import asyncio
from dataclasses import dataclass
from typing import Literal, Union

from maestro import Maestro

from easymesh import build_mesh_node
from easymesh.asyncio import forever

ServoDeg = Union[float, Literal['off'], None]


@dataclass
class ServoCommand:
    pan_deg: ServoDeg = None
    tilt0_deg: ServoDeg = None
    tilt1_deg: ServoDeg = None

    @property
    def pan_us(self) -> float:
        return 0. if self.pan_deg == 'off' else 1500 + self.pan_deg * 500 / 72.3

    @property
    def tilt0_us(self) -> float:
        return 0. if self.tilt0_deg == 'off' else 1500 + self.tilt0_deg * 500 / 45

    @property
    def tilt1_us(self) -> float:
        return 0. if self.tilt1_deg == 'off' else 1500 - self.tilt1_deg * 500 / 45


async def main() -> None:
    node = await build_mesh_node(name='servo')

    print('Connecting to Maestro...')
    with Maestro.connect('mini12', tty='/dev/ttyACM0') as maestro:
        print('Connected!')

        if maestro.script_is_running():
            print('Stopping running script... ', end='', flush=True)
            maestro.stop_script()
            print('Done')

        async def handle_servo_command(topic, command: ServoCommand) -> None:
            print('Received command:', command)

            if command.pan_deg is not None:
                maestro.set_target(0, command.pan_us)
            if command.tilt0_deg is not None:
                maestro.set_target(1, command.tilt0_us)
            if command.tilt1_deg is not None:
                maestro.set_target(2, command.tilt1_us)

        await node.listen('servo_command', handle_servo_command)
        await forever()


if __name__ == '__main__':
    asyncio.run(main())
