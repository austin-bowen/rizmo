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
from rizmo.nodes.messages import ChangeServoPosition, SetHeadSpeed, SetServoPosition
from rizmo.signal import graceful_shutdown_on_sigterm

DEFAULT_TTY = '/dev/ttyACM0'

PAN = 0
TILT0 = 1
TILT1 = 2

SERVOS = (PAN, TILT0, TILT1)

CENTER = 1500.
SPEED = 1000.


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    async def handle_maestro_cmd(topic, command: Union[SetServoPosition, ChangeServoPosition]) -> None:
        print('[Maestro] Received command:', command)

        if isinstance(command, SetServoPosition):
            set_servo_position(command)
        elif isinstance(command, ChangeServoPosition):
            change_servo_position(command)
        elif isinstance(command, SetHeadSpeed):
            set_head_speed(command)
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

    def set_head_speed(command: SetHeadSpeed) -> None:
        runtime = 0.5

        if command.pan_dps is not None:
            speed = command.pan_speed_us_per_second
            maestro.set_speed(PAN, max(abs(speed), 25))
            position = maestro.get_position(PAN)
            maestro[PAN] = position + speed * runtime

        if command.tilt_dps is not None:
            speed = command.tilt_speed_us_per_second
            maestro.set_speed(TILT1, max(abs(speed), 25))
            position = maestro.get_position(TILT1)
            maestro[TILT1] = position + speed * runtime

    def set_servo_speeds(speed: float) -> None:
        for c in SERVOS:
            maestro.set_speed(c, speed)

    def center_servos():
        for c in SERVOS:
            maestro[c] = CENTER

    print(f'Connecting to Maestro on {args.tty!r}...')
    with Maestro.connect('mini12', tty=args.tty, safe_close=False) as maestro:
        print('Connected!')

        # Setup servos
        maestro.stop()
        maestro.set_limits(PAN, 400, 2600)
        maestro.set_limits(TILT0, 512, 2488)
        maestro.set_limits(TILT1, 512, 2208)
        set_servo_speeds(SPEED)
        center_servos()

        await node.listen('maestro_cmd', handle_maestro_cmd)

        try:
            await forever()
        finally:
            set_servo_speeds(250)
            center_servos()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('maestro-ctl')

    parser.add_argument(
        '--tty',
        default=DEFAULT_TTY,
        help="The tty device to connect to the Maestro controller. Default: %(default)s",
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
