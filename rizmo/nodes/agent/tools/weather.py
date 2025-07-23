from dataclasses import asdict

from rosy.node.node import TopicProxy

from rizmo.llm_utils import Tool
from rizmo.weather import WeatherProvider


class GetWeatherTool(Tool):
    def __init__(self, weather_provider: WeatherProvider, say_topic: TopicProxy):
        self.weather_provider = weather_provider
        self.say_topic = say_topic

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='get_weather',
                description=
                'Gets the weather for today, tomorrow, and the week, '
                'as well as the current moon phase.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        location=dict(
                            type='string',
                            description='Weather location, e.g. "New York, NY".',
                        ),
                    ),
                    required=['location'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, location: str) -> dict:
        await self.say_topic.send('Checking...')
        weather = await self.weather_provider.get_weather(location)
        return asdict(weather)
