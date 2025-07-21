/*



*/

import { FeedbackType } from "../../interfaces";
import {
  ChatPacket,
  Packet,
  PacketType,
  ToolPacket,
} from "../../services/streamingModels";
import { FullChatState } from "./interfaces";
import { MessageTextRenderer } from "./renderers/MessageTextRenderer";
import { SearchToolRenderer } from "./renderers/SearchToolRenderer";
import { ImageToolRenderer } from "./renderers/ImageToolRenderer";

// Different types of chat packets using discriminated unions
export interface GroupedPackets {
  packets: Packet[];
}

function isChatPacket(packet: Packet): packet is ChatPacket {
  return (
    packet.obj.type === PacketType.MESSAGE_START ||
    packet.obj.type === PacketType.MESSAGE_DELTA ||
    packet.obj.type === PacketType.MESSAGE_END
  );
}

function isSearchToolPacket(packet: Packet): packet is ToolPacket {
  return (
    packet.obj.type === PacketType.TOOL_START &&
    packet.obj.tool_name === "search"
  );
}

function isImageToolPacket(packet: Packet): packet is ToolPacket {
  return (
    packet.obj.type === PacketType.TOOL_START &&
    packet.obj.tool_name === "image_generation"
  );
}

export function renderMessageComponent(
  groupedPackets: GroupedPackets,
  fullChatState: FullChatState
) {
  if (!groupedPackets.packets || !groupedPackets.packets[0]) {
    return null;
  }

  console.log("groupedPackets", groupedPackets);

  if (groupedPackets.packets.some((packet) => isChatPacket(packet))) {
    return (
      <MessageTextRenderer
        packets={groupedPackets.packets as ChatPacket[]}
        state={fullChatState}
      />
    );
  } else if (
    groupedPackets.packets.some((packet) => isSearchToolPacket(packet))
  ) {
    return (
      <SearchToolRenderer
        packets={groupedPackets.packets as ToolPacket[]}
        state={fullChatState}
      />
    );
  } else if (
    groupedPackets.packets.some((packet) => isImageToolPacket(packet))
  ) {
    return (
      <ImageToolRenderer
        packets={groupedPackets.packets as ToolPacket[]}
        state={fullChatState}
      />
    );
  }
  return null;
}
