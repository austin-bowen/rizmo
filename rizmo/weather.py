import asyncio
from collections import Counter
from collections.abc import Iterable
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

        # Today's forecast

        forecast = weather.daily_forecasts[0]
        high = forecast.highest_temperature
        low = forecast.lowest_temperature

        today = (
            f'It is {weather.temperature} degrees and {weather.description}, '
            f'with a forecasted high of {high}, and a low of {low}.'
        )

        # Tomorrow's forecast

        forecast = weather.daily_forecasts[1]
        kind = most_common(h.kind for h in forecast.hourly_forecasts)
        high = forecast.highest_temperature
        low = forecast.lowest_temperature

        tomorrow = (
            f'Tomorrow, it will be {kind}, with a high of {high}, '
            f'and a low of {low}.'
        )

        # This week's forecast

        kinds = []
        for forecast in weather.daily_forecasts:
            kinds.extend(h.kind for h in forecast.hourly_forecasts)
        kind = most_common(kinds)

        highs = [forecast.highest_temperature for forecast in weather.daily_forecasts]
        high = round(sum(highs) / len(highs))

        lows = [forecast.lowest_temperature for forecast in weather.daily_forecasts]
        low = round(sum(lows) / len(lows))

        this_week = (
            f'This week, it will be {kind}, with highs around {high}, '
            f'and lows around {low}.'
        )

        return Weather(today, tomorrow, this_week)


@dataclass
class Weather:
    today: str
    tomorrow: str
    this_week: str


def most_common(items: Iterable):
    counter = Counter(items)
    return counter.most_common(1)[0][0]


async def main():
    async with WeatherProvider.build(config.weather_location) as weather_provider:
        print(await weather_provider.get_weather())


if __name__ == '__main__':
    asyncio.run(main())
