"""
Limits:
0: 400 - 2600
1: 512 - 2488
2: 512 - 2208
"""

import asyncio
from argparse import Namespace
from typing import Union

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from maestro import Maestro

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import ChangeServoPosition, SetServoPosition
from rizmo.signal import graceful_shutdown_on_sigterm

CENTER = 1500


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    print('Connecting to Maestro...')
    with Maestro.connect('mini12', tty='/dev/ttyACM0') as maestro:
        print('Connected!')

        maestro.stop()
        maestro.set_limits(0, 400, 2600)
        maestro.set_limits(1, 512, 2488)
        maestro.set_limits(2, 512, 2208)
        for c in range(3):
            maestro.set_speed(c, 40)
            maestro[c] = CENTER

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
                maestro[1] = min(max(CENTER, maestro[1] + command.tilt0_us), 1750)
            if command.tilt1_deg is not None:
                maestro[2] = min(max(0, maestro[2] + command.tilt1_us), 4090)

        await node.listen('maestro_cmd', handle_maestro_cmd)

        try:
            await forever()
        finally:
            await node.stop_listening('maestro_cmd')

            for c in range(3):
                maestro[c] = CENTER

            await asyncio.to_thread(maestro.wait_until_done_moving)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('maestro-ctl')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
