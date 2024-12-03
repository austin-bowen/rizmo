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

PAN = 0
TILT0 = 1
TILT1 = 2

SERVOS = (PAN, TILT0, TILT1)

CENTER = 1500.
SPEED = 40


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    print('Connecting to Maestro...')
    with Maestro.connect('mini12', tty='/dev/ttyACM0', safe_close=False) as maestro:
        print('Connected!')

        def set_servo_speeds(speed: int) -> None:
            for c in SERVOS:
                maestro.set_speed(c, speed)

        def center_servos():
            for c in SERVOS:
                maestro[c] = CENTER

        maestro.stop()
        maestro.set_limits(PAN, 400, 2600)
        maestro.set_limits(TILT0, 512, 2488)
        maestro.set_limits(TILT1, 512, 2208)
        set_servo_speeds(SPEED)
        center_servos()

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
                maestro[PAN] = command.pan_us
            if command.tilt0_deg is not None:
                maestro[TILT0] = command.tilt0_us
            if command.tilt1_deg is not None:
                maestro[TILT1] = command.tilt1_us

        def change_servo_position(command: ChangeServoPosition):
            if command.pan_deg is not None:
                maestro[PAN] = min(max(0., maestro.get_position(PAN) + command.pan_us), 4090.)
            if command.tilt0_deg is not None:
                maestro[TILT0] = min(max(CENTER, maestro.get_position(TILT0) + command.tilt0_us), 1750.)
            if command.tilt1_deg is not None:
                maestro[TILT1] = min(max(0., maestro.get_position(TILT1) + command.tilt1_us), 4090.)

        await node.listen('maestro_cmd', handle_maestro_cmd)

        try:
            await forever()
        finally:
            set_servo_speeds(10)
            center_servos()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('maestro-ctl')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
