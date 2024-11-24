import asyncio

from easymesh import build_mesh_node
from .servo_node import ServoCommand


def get_servo_command() -> ServoCommand:
    commands = input('<channel> <target_deg>[,] | off: ')
    if not commands:
        return ServoCommand()

    if commands == 'off':
        return ServoCommand('off', 'off', 'off')

    servo_command = ServoCommand()

    for command in commands.split(','):
        channel, target_deg = command.strip().lower().split()

        channel = int(channel)

        if target_deg != 'off':
            target_deg = float(target_deg)

        if channel == 0:
            servo_command.pan_deg = target_deg
        elif channel == 1:
            servo_command.tilt0_deg = target_deg
        elif channel == 2:
            servo_command.tilt1_deg = target_deg

    return servo_command


async def main() -> None:
    node = await build_mesh_node(name='servo')

    while True:
        servo_command = await asyncio.to_thread(get_servo_command)
        await node.send('servo_command', servo_command)


if __name__ == '__main__':
    asyncio.run(main())
