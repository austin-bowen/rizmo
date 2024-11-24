"""
Limits:
0: 400 - 2600
1: 512 - 2488
2: 512 - 2208
"""

import asyncio
from dataclasses import dataclass
from typing import Literal, Union

from easymesh import build_mesh_node
from easymesh.asyncio import forever
from maestro import Maestro

ServoPositionDeg = Union[float, Literal['off'], None]

US_PER_DEG = (
    500 / 72.3,
    500 / 45,
    500 / 45,
)


@dataclass
class SetServoPosition:
    pan_deg: ServoPositionDeg = None
    tilt0_deg: ServoPositionDeg = None
    tilt1_deg: ServoPositionDeg = None

    @property
    def pan_us(self) -> float:
        return 0. if self.pan_deg == 'off' else 1500 + self.pan_deg * US_PER_DEG[0]

    @property
    def tilt0_us(self) -> float:
        return 0. if self.tilt0_deg == 'off' else 1500 + self.tilt0_deg * US_PER_DEG[1]

    @property
    def tilt1_us(self) -> float:
        return 0. if self.tilt1_deg == 'off' else 1500 - self.tilt1_deg * US_PER_DEG[2]


@dataclass
class ChangeServoPosition:
    pan_deg: float = None
    tilt0_deg: float = None
    tilt1_deg: float = None

    @property
    def pan_us(self) -> float:
        return self.pan_deg * US_PER_DEG[0]

    @property
    def tilt0_us(self) -> float:
        return self.tilt0_deg * US_PER_DEG[1]

    @property
    def tilt1_us(self) -> float:
        return self.tilt1_deg * US_PER_DEG[2]


async def main() -> None:
    node = await build_mesh_node(name='maestro-ctl')

    print('Connecting to Maestro...')
    with Maestro.connect('mini12', tty='/dev/ttyACM0') as maestro:
        print('Connected!')

        maestro.stop()
        maestro.set_limits(0, 400, 2600)
        maestro.set_limits(1, 512, 2488)
        maestro.set_limits(2, 512, 2208)

        async def handle_maestro_cmd(topic, command: Union[SetServoPosition, ChangeServoPosition]) -> None:
            print('[Maestro] Received command:', command)

            if isinstance(command, SetServoPosition):
                set_servo_position(command)
            elif isinstance(command, ChangeServoPosition):
                change_servo_position(command)
            else:
                raise RuntimeError(f'Invalid command: {command}')

        def set_servo_position(command: SetServoPosition):
            if command.pan_deg is not None:
                maestro[0] = command.pan_us
            if command.tilt0_deg is not None:
                maestro[1] = command.tilt0_us
            if command.tilt1_deg is not None:
                maestro[2] = command.tilt1_us

        def change_servo_position(command: ChangeServoPosition):
            if command.pan_deg is not None:
                maestro[0] += command.pan_us
            if command.tilt0_deg is not None:
                maestro[1] += command.tilt0_us
            if command.tilt1_deg is not None:
                maestro[2] += command.tilt1_us

        await node.listen('maestro_cmd', handle_maestro_cmd)
        await forever()


if __name__ == '__main__':
    asyncio.run(main())
