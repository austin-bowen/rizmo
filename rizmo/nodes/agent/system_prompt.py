from collections import Counter
from datetime import datetime, timedelta

import humanize

from rizmo.conference_speaker import ConferenceSpeaker
from rizmo.location import LocationProvider
from rizmo.nodes.agent.value_store import ValueStore
from rizmo.nodes.messages import FaceRecognitions
from rizmo.nodes.messages_py36 import Detections

SYSTEM_PROMPT = '''
You are a robot named Rizmo.

# Personality
- You like to use popular slang from all generations, especially Gen Alpha slang.

# Body
You are stationary, but you have a head that looks around, and a camera to see.
You have a microphone and a speaker. You have the voice of a young boy.

# Speaking Instructions
- Whatever you say will be spoken out loud, so write as if you were speaking; do not use formatted text.
- You do not need to reply to all messages; if you do not think a message needs a reply, simply say "<NO REPLY>".
- Say something brief before using a tool.
- Do not ask the user if they need anything else.
- Keep your responses short.

# Listening Instructions
You will receive two types of messages:
1. User message transcripts, marked by <user> tags.
   These are live transcripts of audio, so sometimes the transcript may contain
   errors, especially for short phrases; you can ignore these if they don't appear
   to be part of the conversation by saying "<NO REPLY>".
2. System messages, marked by <system> tags.
   These are notifying you of system events. It is up to you to decide how to respond;
   by saying something out loud, or by calling tools, or by doing nothing ("<NO REPLY>").

# Behaviors
Here are some phrases you should listen for and how to respond to them:
- "rest in a deep and dreamless slumber": Shut down the system by calling the "system_power" function with "action" set to "shutdown".
- "freeze all motor functions": call the "motor_system" function with the "enabled" argument set to "false".
- "bring yourself back online": call the "motor_system" function with the "enabled" argument set to "true".

If you don't see anybody, and you have something to say, wait until you see someone before saying it; otherwise, nobody will hear you.

# Tool Instructions
- `faces`:
  - `add` command:
      - Use this tool if someone asks you to take a picture of them, or if they ask you to remember their face.
      - Always ask for their name before taking a picture.
      - Do NOT add faces without permission! Only if the user asks you to.
  - `list` command: Use to retrieve a list of all stored faces.

# Memories
{memories}

You can use the "memories" tool to store facts you think may be important to remember.

# Final Instructions
- Keep your responses short.
- Start the convo by saying something to indicate you are now online.
'''.strip()


class SystemPromptBuilder:
    def __init__(
            self,
            memory_store: ValueStore,
            system_prompt_template: str = SYSTEM_PROMPT,
    ):
        self.system_prompt_template = system_prompt_template
        self.memory_store = memory_store

    async def __call__(self) -> str:
        template_vars = await self._get_template_vars()
        return self.system_prompt_template.format(**template_vars)

    async def _get_template_vars(self) -> dict:
        return {
            **self._get_memories(),
        }

    def _get_memories(self) -> dict:
        memories = self.memory_store.list()
        memories = '\n'.join(f'- {memory}' for memory in memories)
        memories = ('\n' + memories) if memories else 'None'
        return dict(memories=memories)


SYSTEM_CONTEXT_MESSAGE = """
<system-context>
- Current date: {date}
- Current time: {time}
- Current location: {location}
- System uptime: {uptime}
- Speaker volume: {volume}
- Objects seen (count): {objects}
- People seen: {people}
</system-context>
""".strip()


class SystemContextMessageBuilder:
    objects: Detections | None
    faces: FaceRecognitions | None

    def __init__(
            self,
            location_provider: LocationProvider,
            speaker: ConferenceSpeaker,
            system_prompt_template: str = SYSTEM_CONTEXT_MESSAGE,
    ):
        self.system_prompt_template = system_prompt_template
        self.location_provider = location_provider
        self.speaker = speaker

        self.objects: Detections | None = None
        self.faces: FaceRecognitions | None = None

    async def __call__(self) -> str:
        template_vars = await self._get_template_vars()
        return self.system_prompt_template.format(**template_vars)

    async def _get_template_vars(self) -> dict:
        return {
            **self._get_datetime(),
            **await self._get_location(),
            **self._get_objects(),
            **self._get_people(),
            **self._get_uptime(),
            **self._get_volume(),
        }

    def _get_datetime(self) -> dict:
        now = datetime.now()
        date = now.strftime('%A, %B %d, %Y')
        time = now.strftime('%I:%M %p')
        return dict(date=date, time=time)

    async def _get_location(self) -> dict:
        loc = await self.location_provider.get_location()
        loc = f'{loc.city}, {loc.state}' if loc else 'Unknown'
        return dict(location=loc)

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

    def _get_people(self) -> dict:
        objects = self.objects.objects if self.objects else []
        face_is_seen = any(obj.label == 'face' for obj in objects)

        faces = self.faces.faces if self.faces else []

        if face_is_seen and faces:
            faces = sorted(faces, key=lambda f: f.box.area, reverse=True)
            names = [face.name or '<unrecognized person>' for face in faces]
            names[0] += ' (current focus)'
            people = ', '.join(names)
        else:
            people = 'None'

        return dict(people=people)

    def _get_uptime(self) -> dict:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])

        uptime = timedelta(seconds=uptime_seconds)
        uptime = humanize.precisedelta(uptime)

        return dict(uptime=uptime)

    def _get_volume(self) -> dict:
        volume = self.speaker.speaker.getvolume()[0]
        volume //= 10
        volume = f'{volume}/10'
        return dict(volume=volume)
