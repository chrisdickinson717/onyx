from enum import Enum
from typing import Literal

from pydantic import BaseModel

from onyx.chat.models import LlmDoc


class PacketType(str, Enum):
    # Basic Message Packets
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"

    # Control Packets
    STOP = "stop"

    # Tool Packets
    SEARCH_TOOL_START = "search_tool_start"
    SEARCH_TOOL_END = "search_tool_end"
    IMAGE_TOOL_START = "image_tool_start"
    IMAGE_TOOL_END = "image_tool_end"


class BaseObj(BaseModel):
    type: str


"""Basic Message Packets"""


class MessageStart(BaseObj):
    id: str
    type: Literal[PacketType.MESSAGE_START] = PacketType.MESSAGE_START
    content: str


class MessageDelta(BaseObj):
    content: str
    type: Literal[PacketType.MESSAGE_DELTA] = PacketType.MESSAGE_DELTA


class MessageEnd(BaseObj):
    type: Literal[PacketType.MESSAGE_END] = PacketType.MESSAGE_END


"""Control Packets"""


class Stop(BaseObj):
    type: Literal[PacketType.STOP] = PacketType.STOP


"""Tool Packets"""


class SearchToolStart(BaseObj):
    type: Literal[PacketType.SEARCH_TOOL_START] = PacketType.SEARCH_TOOL_START
    query: str


class SearchToolEnd(BaseObj):
    type: Literal[PacketType.SEARCH_TOOL_END] = PacketType.SEARCH_TOOL_END
    results: list[LlmDoc]


class ImageToolStart(BaseObj):
    type: Literal[PacketType.IMAGE_TOOL_START] = PacketType.IMAGE_TOOL_START
    prompt: str


class ImageToolEnd(BaseObj):
    type: Literal[PacketType.IMAGE_TOOL_END] = PacketType.IMAGE_TOOL_END
    images: list[dict[str, str]]  # List of {id, url, prompt} objects


ObjTypes = (
    MessageStart
    | MessageDelta
    | MessageEnd
    | Stop
    | SearchToolStart
    | SearchToolEnd
    | ImageToolStart
    | ImageToolEnd
)


class Packet(BaseModel):
    ind: int
    obj: ObjTypes
