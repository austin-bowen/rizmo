from rizmo.llm_utils import Tool
from rizmo.nodes.agent.value_store import ValueStore


class MemoryTool(Tool):
    def __init__(self, memory_store: ValueStore):
        self.memory_store = memory_store

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='memories',
                description='Manages memories.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        action=dict(
                            type='string',
                            description='The action to perform.',
                            enum=['add', 'remove'],
                        ),
                        memory=dict(
                            type='string',
                            description='The memory to add/remove.',
                        ),
                    ),
                    required=['action', 'memory'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: str, memory: str) -> str:
        if action == 'add':
            self.memory_store.add(memory)
            return f'Added memory: "{memory}"'
        elif action == 'remove':
            self.memory_store.remove(memory)
            return f'Removed memory: "{memory}"'
        else:
            raise ValueError(f'Invalid action: {action}')
