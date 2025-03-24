from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import aiohttp


def get_location_provider() -> 'LocationProvider':
    return CachedLocationProvider(
        LocationProviders([
            IPLocationProvider(),
        ])
    )


@dataclass
class Location:
    city: str
    state: str


class LocationProvider(ABC):
    @abstractmethod
    async def get_location(self) -> Optional[Location]:
        ...


class LocationProviders(LocationProvider):
    def __init__(self, providers: Iterable[LocationProvider]):
        self.providers = providers

    async def get_location(self) -> Optional[Location]:
        for provider in self.providers:
            location = await provider.get_location()
            if location is not None:
                return location

        return None


class CachedLocationProvider(LocationProvider):
    def __init__(self, provider: LocationProvider):
        self.provider = provider
        self.location = None

    async def get_location(self) -> Optional[Location]:
        if self.location is None:
            self.location = await self.provider.get_location()

        return self.location


class IPLocationProvider(LocationProvider):
    async def get_location(self) -> Optional[Location]:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://ipinfo.io/json') as response:
                data = await response.json()

        city = data.get('city')
        state = data.get('region')

        return Location(city=city, state=state) if city and state else None


async def _main() -> None:
    location_provider = get_location_provider()
    location = await location_provider.get_location()
    print('Location:', location)


if __name__ == '__main__':
    import asyncio

    asyncio.run(_main())
