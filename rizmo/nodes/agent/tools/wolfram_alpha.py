from typing import Optional

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

        result = self._get_result(response)
        if result:
            return result

        did_you_mean = self._get_did_you_mean(response)
        if did_you_mean:
            return f'Did you mean: {did_you_mean}'

        print(f'ERROR response: {response}')
        return 'ERROR: No response.'

    def _get_result(self, response: wolframalpha.Result) -> Optional[str]:
        result = next(response.results, None)
        return result.text if result else None

    def _get_did_you_mean(self, response: wolframalpha.Result) -> Optional[str]:
        did_you_mean = response.get('didyoumeans', None)
        if not did_you_mean:
            return None

        did_you_mean = did_you_mean.get('didyoumean', None)
        if not did_you_mean:
            return None

        return did_you_mean.get('@val', None)
