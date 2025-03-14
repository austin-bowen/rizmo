import asyncio
import subprocess
from asyncio import subprocess
from collections.abc import Sequence


class NetworkManager:
    async def connect(self, name: str) -> None:
        await self._run_and_check_command([
            "nmcli",
            "connection",
            "up",
            name,
        ])

    async def connection_is_active(self, name: str) -> bool:
        return name in await self.get_active_connections()

    async def get_active_connections(self):
        result = await self._run_and_check_command([
            "nmcli",
            "--terse",
            "--fields", "NAME",
            "connection", "show",
            "--active",
        ])

        return result.splitlines()

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
