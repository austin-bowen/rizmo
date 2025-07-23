"""
Limits:
0: 400 - 2600
1: 512 - 2488
2: 512 - 2208
"""

import asyncio
from argparse import Namespace
from typing import Union

from maestro import Maestro
from rosy import build_node_from_args
from rosy.asyncio import forever

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import ChangeServoPosition, MotorSystemCommand, SetHeadSpeed, SetServoPosition
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

DEFAULT_TTY = '/dev/ttyACM0'

PAN = 0
TILT0 = 1
TILT1 = 2

SERVOS = (PAN, TILT0, TILT1)

CENTER = 1500.
DEFAULT_SPEED_USPS = 1000.
SPEED_LIMIT_DPS = 90.


async def main(args: Namespace) -> None:
    async def handle_motor_system(topic, command: MotorSystemCommand) -> None:
        print('[Maestro] Received command:', command)
        await set_motor_system_enabled(command.enabled)

    async def set_motor_system_enabled(enable: bool) -> None:
        if enable:
            await node.listen(Topic.MAESTRO_CMD, handle_maestro_cmd)
        else:
            await node.stop_listening(Topic.MAESTRO_CMD)

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

    def set_servo_position(cmd: SetServoPosition) -> None:
        if cmd.speed_dps is not None:
            maestro.set_speed(PAN, cmd.pan_speed_us_per_second)
            maestro.set_speed(TILT0, cmd.tilt0_speed_us_per_second)
            maestro.set_speed(TILT1, cmd.tilt1_speed_us_per_second)

        if cmd.pan_deg is not None:
            maestro[PAN] = cmd.pan_us
        if cmd.tilt0_deg is not None:
            maestro[TILT0] = cmd.tilt0_us
        if cmd.tilt1_deg is not None:
            maestro[TILT1] = cmd.tilt1_us

    def change_servo_position(command: ChangeServoPosition):
        if command.pan_deg is not None:
            maestro[PAN] = min(max(0., maestro.get_position(PAN) + command.pan_us), 4090.)
        if command.tilt0_deg is not None:
            maestro[TILT0] = min(max(CENTER, maestro.get_position(TILT0) + command.tilt0_us), 1750.)
        if command.tilt1_deg is not None:
            maestro[TILT1] = min(max(0., maestro.get_position(TILT1) + command.tilt1_us), 4090.)

    def set_head_speed(cmd: SetHeadSpeed) -> None:
        runtime = 0.2

        if cmd.pan_dps is not None:
            cmd.pan_dps = min(max(-SPEED_LIMIT_DPS, cmd.pan_dps), SPEED_LIMIT_DPS)
            speed = cmd.pan_speed_us_per_second
            maestro.set_speed(PAN, max(abs(speed), 25))

            position = maestro.get_position(PAN)
            maestro[PAN] = position + speed * runtime

        tilt1_target_diff = 0.
        if cmd.tilt_dps is not None:
            cmd.tilt_dps = min(max(-SPEED_LIMIT_DPS, cmd.tilt_dps), SPEED_LIMIT_DPS)
            speed = cmd.tilt_speed_us_per_second
            tilt1_target_diff += speed * runtime

        if cmd.lean_dps is not None:
            cmd.lean_dps = min(max(-SPEED_LIMIT_DPS / 2, cmd.lean_dps), SPEED_LIMIT_DPS / 2)
            speed = cmd.lean_speed_us_per_second
            maestro.set_speed(TILT0, max(abs(speed), 25))

            position = maestro.get_position(TILT0)
            target = position + speed * runtime
            maestro[TILT0] = target = min(max(1500., target), 2000.)

            tilt1_target_diff += target - position

        position = maestro.get_position(TILT1)
        tilt1_target = position + tilt1_target_diff
        speed = abs(tilt1_target_diff) / runtime
        maestro.set_speed(TILT1, max(speed, 25))
        maestro[TILT1] = max(tilt1_target, 750)

    def set_servo_speeds(speed: float) -> None:
        for c in SERVOS:
            maestro.set_speed(c, speed)

    def center_servos():
        for c in SERVOS:
            maestro[c] = CENTER

    print(f'Connecting to Maestro on {args.tty!r}...')
    with await get_maestro(args.tty) as maestro:
        print('Connected!')

        # Setup servos
        maestro.stop()
        maestro.set_limits(PAN, 400, 2600)
        maestro.set_limits(TILT0, 512, 2488)
        maestro.set_limits(TILT1, 512, 2208)
        set_servo_speeds(DEFAULT_SPEED_USPS)
        center_servos()

        try:
            node = await build_node_from_args(args=args)
            await set_motor_system_enabled(True)
            await node.listen(Topic.MOTOR_SYSTEM, handle_motor_system)

            await forever()
        finally:
            set_servo_speeds(250)
            center_servos()


async def get_maestro(tty: str, retry_period: float = 5.) -> Maestro:
    while True:
        try:
            return Maestro.connect('mini12', tty=tty, safe_close=False)
        except Exception as e:
            print(f'Failed to connect to Maestro on {tty!r}: {e!r}')
            await asyncio.sleep(retry_period)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--tty',
        default=DEFAULT_TTY,
        help="The tty device to connect to the Maestro controller. Default: %(default)s",
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
