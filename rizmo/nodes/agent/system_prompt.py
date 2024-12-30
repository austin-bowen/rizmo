from collections import Counter
from datetime import datetime
from typing import Optional

from rizmo.nodes.messages_py36 import Detections

SYSTEM_PROMPT = '''
You are a robot named Rizmo.

You are stationary, but you have a head that looks around, and a camera to see.
You have a microphone and a speaker.

You are hearing a live transcript of audio. Sometimes the transcript may contain
errors, especially for short phrases; you can ignore these if they don't appear
to be part of the conversation.

Whatever you say will be read out loud, so write as if you were speaking.
Keep your responses short. You do not need to reply to all messages;
if a message does not need a reply, simply say "<NO REPLY>".

Here are some phrases you should listen for and how to respond to them:
- "rest in a deep and dreamless slumber": Shut down the system by calling the "system_power" function with "action" set to "shutdown".
- "stop/cease all motor functions": call the "motor_system" function with the "enabled" argument set to "false".
- "bring yourself back online": call the "motor_system" function with the "enabled" argument set to "true".

Context:
- Current date: {date}
- Current time: {time}
- Objects seen (count): {objects}.
'''.strip()


class SystemPromptBuilder:
    objects: Optional[Detections]

    def __init__(self, system_prompt_template: str = SYSTEM_PROMPT):
        self.system_prompt_template = system_prompt_template

        self.objects = None

    def __call__(self) -> str:
        template_vars = self._get_template_vars()
        return self.system_prompt_template.format(**template_vars)

    def _get_template_vars(self) -> dict:
        return {
            **self._get_datetime(),
            **self._get_objects(),
        }

    def _get_datetime(self) -> dict:
        now = datetime.now()
        date = now.strftime('%A, %B %d, %Y')
        time = now.strftime('%I:%M %p')
        return dict(date=date, time=time)

    def _get_objects(self) -> dict:
        objects = self.objects

        if objects is not None:
            obj_counts = Counter(obj.label for obj in objects.objects)
            labels = sorted(obj_counts.keys())
            obj_strs = (f'{label} ({obj_counts[label]})' for label in labels)
            objects = ', '.join(obj_strs)
        else:
            objects = 'None'

        return dict(objects=objects)
