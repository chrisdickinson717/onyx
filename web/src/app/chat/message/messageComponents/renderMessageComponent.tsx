/*



*/

import { FeedbackType } from "../../interfaces";
import {
  ChatPacket,
  Packet,
  PacketType,
  SearchToolPacket,
} from "../../services/streamingModels";
import { FullChatState } from "./interfaces";
import { MessageTextRenderer } from "./MessageTextRenderer";
import { SearchToolRenderer } from "./SearchToolRenderer";

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

function isSearchToolPacket(packet: Packet): packet is SearchToolPacket {
  return (
    packet.obj.type === PacketType.SEARCH_TOOL_START ||
    packet.obj.type === PacketType.SEARCH_TOOL_END
  );
}

export function renderMessageComponent(
  groupedPackets: GroupedPackets,
  fullChatState: FullChatState
) {
  if (!groupedPackets.packets || !groupedPackets.packets[0]) {
    return null;
  }

  if (isChatPacket(groupedPackets.packets[0])) {
    return (
      <MessageTextRenderer
        packets={groupedPackets.packets as ChatPacket[]}
        state={fullChatState}
      />
    );
  } else if (isSearchToolPacket(groupedPackets.packets[0])) {
    return (
      <SearchToolRenderer
        packets={groupedPackets.packets as SearchToolPacket[]}
        state={fullChatState}
      />
    );
  }
  return null;
}
