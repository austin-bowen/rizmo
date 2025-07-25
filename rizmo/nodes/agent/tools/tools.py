import wolframalpha
from rosy import Node

from rizmo import secrets
from rizmo.conference_speaker import ConferenceSpeaker
from rizmo.config import config
from rizmo.llm_utils import ToolHandler
from rizmo.nodes.agent.tools.faces import FacesTool
from rizmo.nodes.agent.tools.memory import MemoryTool
from rizmo.nodes.agent.tools.motor_system import MotorSystemTool
from rizmo.nodes.agent.tools.reminders import RemindersTool
from rizmo.nodes.agent.tools.system_power import SystemPowerTool
from rizmo.nodes.agent.tools.system_status import GetSystemStatusTool
from rizmo.nodes.agent.tools.timer import TimerTool
from rizmo.nodes.agent.tools.volume import VolumeTool
from rizmo.nodes.agent.tools.weather import GetWeatherTool
from rizmo.nodes.agent.tools.wolfram_alpha import WolframAlphaTool
from rizmo.nodes.agent.value_store import ValueStore
from rizmo.nodes.topics import Topic
from rizmo.weather import WeatherProvider


def get_tool_handler(
        node: Node,
        memory_store: ValueStore,
        timer_complete_callback,
        speaker: ConferenceSpeaker,
) -> ToolHandler:
    face_cmd_topic = node.get_topic(Topic.FACE_COMMAND)
    say_topic = node.get_topic(Topic.SAY)
    weather_provider = WeatherProvider.build()
    reminder_store = ValueStore(config.reminders_file_path)
    wa_client = wolframalpha.Client(secrets.WOLFRAM_ALPHA_APP_ID)

    return ToolHandler([
        FacesTool(face_cmd_topic),
        GetSystemStatusTool(say_topic),
        GetWeatherTool(weather_provider, say_topic),
        MemoryTool(memory_store),
        MotorSystemTool(node.get_topic(Topic.MOTOR_SYSTEM)),
        RemindersTool(reminder_store),
        SystemPowerTool(say_topic),
        TimerTool(timer_complete_callback),
        VolumeTool(speaker),
        WolframAlphaTool(wa_client),
    ])
