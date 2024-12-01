import asyncio
from dataclasses import dataclass

import python_weather

from rizmo.config import config


class WeatherProvider:
    def __init__(self, client: python_weather.Client, location: str):
        self.client = client
        self.location = location

    @classmethod
    def build(cls, location: str) -> 'WeatherProvider':
        client = python_weather.Client(unit=python_weather.IMPERIAL)
        return cls(client, location)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self) -> None:
        await self.client.close()

    async def get_weather(self) -> 'Weather':
        weather = await self.client.get(self.location)

        forecast = next(iter(weather))
        high = forecast.highest_temperature
        low = forecast.lowest_temperature

        today = (
            f'It is {weather.temperature} degrees and {weather.kind}, '
            f'with a forecasted high of {high}, and a low of {low}.'
        )

        # TODO
        tomorrow = f'Tomorrow, {today}'
        this_week = f'This week, {today}'

        return Weather(today, tomorrow, this_week)


@dataclass
class Weather:
    today: str
    tomorrow: str
    this_week: str


async def main():
    async with WeatherProvider.build(config.weather_location) as weather_provider:
        print(await weather_provider.get_description())


if __name__ == '__main__':
    asyncio.run(main())
