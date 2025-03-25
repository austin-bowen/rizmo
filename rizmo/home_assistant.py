from homeassistant_api import Client as HomeAssistantClient

from rizmo import secrets
from rizmo.config import config


class HomeAssistant:
    def __init__(self, client: HomeAssistantClient):
        self.client = client

    @classmethod
    def build(
            cls,
            api_url: str = config.home_assistant_api_url,
            token: str = secrets.HOME_ASSISTANT_ACCESS_TOKEN,
    ) -> 'HomeAssistant':
        client = HomeAssistantClient(api_url, token, use_async=True)
        return cls(client)

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def get_states(self):
        return await self.client.async_get_states()


async def _main():
    async with HomeAssistant.build() as ha:
        states = await ha.get_states()
        print('States:')
        for i, state in enumerate(states):
            print(f'\n{i}. {state}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(_main())
