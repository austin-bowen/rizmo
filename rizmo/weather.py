import asyncio
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import python_weather
from python_weather.constants import _Unit as Unit


class WeatherProvider:
    def __init__(self, client: python_weather.Client):
        self.client = client

    @classmethod
    def build(cls, unit: Unit = python_weather.IMPERIAL) -> 'WeatherProvider':
        client = python_weather.Client(unit=unit)
        return cls(client)

    @property
    def unit(self) -> Unit:
        return self.client.unit

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self) -> None:
        await self.client.close()

    async def get_weather(self, location: str) -> 'Weather':
        weather = await self.client.get(location)
        unit = self.unit.temperature

        # Today's forecast

        forecast = weather.daily_forecasts[0]
        high = forecast.highest_temperature
        low = forecast.lowest_temperature

        today = (
            f'It is {weather.temperature}{unit} and {weather.description}, '
            f'with a forecasted high of {high}{unit}, and a low of {low}{unit}.'
        )

        # Tomorrow's forecast

        forecast = weather.daily_forecasts[1]
        kind = most_common(h.kind for h in forecast.hourly_forecasts)
        high = forecast.highest_temperature
        low = forecast.lowest_temperature

        tomorrow = (
            f'Tomorrow, it will be {kind}, with a high of {high}{unit}, '
            f'and a low of {low}{unit}.'
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
            f'This week, it will be {kind}, with highs around {high}{unit}, '
            f'and lows around {low}{unit}.'
        )

        moon_phase = weather.daily_forecasts[0].moon_phase.value

        return Weather(today, tomorrow, this_week, moon_phase, self.unit.temperature)


@dataclass
class Weather:
    today: str
    tomorrow: str
    this_week: str
    moon_phase: str
    temp_unit: str


def most_common(items: Iterable):
    counter = Counter(items)
    return counter.most_common(1)[0][0]


async def main():
    import sys

    location = sys.argv[1]

    async with WeatherProvider.build() as weather_provider:
        print(await weather_provider.get_weather(location))


if __name__ == '__main__':
    asyncio.run(main())
