import asyncio

import python_weather


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

    async def get_description(self) -> str:
        weather = await self.client.get(self.location)

        forecast = next(iter(weather))
        high = forecast.highest_temperature
        low = forecast.lowest_temperature

        return (
            f'It is {weather.temperature} degrees and {weather.kind}, '
            f'with a forecasted high of {high}, and a low of {low}.'
        )


async def main():
    async with WeatherProvider.build('Anderson, SC') as weather_provider:
        print(await weather_provider.get_description())


if __name__ == '__main__':
    asyncio.run(main())
