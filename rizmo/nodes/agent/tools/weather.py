from dataclasses import asdict

from easymesh.node.node import TopicSender

from rizmo.llm_utils import Tool
from rizmo.weather import WeatherProvider


class GetWeatherTool(Tool):
    def __init__(self, weather_provider: WeatherProvider, say_topic: TopicSender):
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
            ),
        )

    async def call(self) -> dict:
        await self.say_topic.send('Checking...')
        weather = await self.weather_provider.get_weather()
        return asdict(weather)
