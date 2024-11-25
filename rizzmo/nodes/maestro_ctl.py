"""
Limits:
0: 400 - 2600
1: 512 - 2488
2: 512 - 2208
"""

import asyncio
from typing import Union

from easymesh import build_mesh_node
from easymesh.asyncio import forever
from maestro import Maestro

from rizzmo.nodes.messages import ChangeServoPosition, SetServoPosition


async def main() -> None:
    node = await build_mesh_node(name='maestro-ctl')

    print('Connecting to Maestro...')
    with Maestro.connect('mini12', tty='/dev/ttyACM0') as maestro:
        print('Connected!')

        maestro.stop()
        maestro.set_limits(0, 400, 2600)
        maestro.set_limits(1, 512, 2488)
        maestro.set_limits(2, 512, 2208)
        for c in range(3):
            maestro.set_speed(c, 40)
            maestro[c] = 1500

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
                maestro[0] = min(max(0, maestro[0] + command.pan_us), 4090)
            if command.tilt0_deg is not None:
                maestro[1] = min(max(1500 - 250, maestro[1] + command.tilt0_us), 1750)
            if command.tilt1_deg is not None:
                maestro[2] = min(max(0, maestro[2] + command.tilt1_us), 4090)

        await node.listen('maestro_cmd', handle_maestro_cmd)
        await forever()


if __name__ == '__main__':
    asyncio.run(main())
