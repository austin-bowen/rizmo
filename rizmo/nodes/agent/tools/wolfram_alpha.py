import wolframalpha

from rizmo.llm_utils import Tool


class WolframAlphaTool(Tool):
    def __init__(self, client: wolframalpha.Client):
        self.client = client

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='wolfram_alpha',
                description='Calls WolframAlpha. Should be used to answer any math questions.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        query=dict(
                            type='string',
                            description='The query to send to WolframAlpha.',
                        ),
                    ),
                    required=['query'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, query: str) -> str:
        response = await self.client.aquery(query)
        return next(response.results).text
