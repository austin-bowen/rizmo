import asyncio
import subprocess
from asyncio import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


class NetworkManager:
    async def cycle_wifi_radio_powered(self, wait_time: float = 5.) -> None:
        await self.set_wifi_radio_powered(False)
        await asyncio.sleep(wait_time)
        await self.set_wifi_radio_powered(True)

    async def set_wifi_radio_powered(self, powered: bool) -> None:
        await self._run_and_check_command([
            "sudo",
            "nmcli",
            "radio",
            "wifi",
            "on" if powered else "off",
        ])

    async def connect(self, name: str) -> None:
        await self._run_and_check_command([
            "nmcli",
            "connection",
            "up",
            name,
        ])

    async def device_is_connected(self, device: str) -> bool:
        connections = await self.get_active_connections()
        return any(conn.device == device for conn in connections)

    async def get_active_connections(self) -> list['Connection']:
        result = await self._run_and_check_command([
            "nmcli",
            "--terse",
            "--fields", "DEVICE,NAME",
            "connection", "show",
            "--active",
        ])

        conns = []
        for line in result.splitlines():
            device, name = line.split(":", maxsplit=1)
            conns.append(Connection(device, name))

        return conns

    async def _run_and_check_command(self, args: Sequence[str]) -> str:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {process.returncode}. "
                f"command={args}; stdout={stdout}; stderr={stderr}"
            )

        return stdout.decode()


@dataclass
class Connection:
    device: str
    name: str
