from rizmo.llm_utils import Tool
from rizmo.nodes.agent.value_store import ValueStore


class RemindersTool(Tool):
    def __init__(self, reminder_store: ValueStore):
        self.reminder_store = reminder_store

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='reminders',
                description='Manages the reminders system and returns the list of reminders after the action is performed.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        action=dict(
                            type='string',
                            description='The action to perform.',
                            enum=['list', 'add', 'remove', 'clear'],
                        ),
                        reminder=dict(
                            type='string',
                            description='The reminder to add/remove. Ignored if action is "list" or "clear"',
                        ),
                    ),
                    required=['action'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: str, reminder: str = None) -> list[str]:
        if action == 'list':
            pass
        elif action == 'add':
            self.reminder_store.add(reminder)
        elif action == 'remove':
            self.reminder_store.remove(reminder)
        elif action == 'clear':
            self.reminder_store.clear()
        else:
            raise ValueError(f'Invalid action: {action}')

        return self.reminder_store.list()
