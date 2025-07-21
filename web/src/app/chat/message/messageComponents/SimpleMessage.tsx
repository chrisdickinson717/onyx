import {
  Packet,
  MessageStart,
  MessageDelta,
  PacketType,
} from "../../services/streamingModels";
import { FullChatState } from "./interfaces";
import { renderMessageComponent } from "./renderMessageComponent";
import { AssistantIcon } from "@/components/assistants/AssistantIcon";
import { CopyButton } from "@/components/CopyButton";
import { LikeFeedback, DislikeFeedback } from "@/components/icons/icons";
import { HoverableIcon } from "@/components/Hoverable";
import {
  CustomTooltip,
  TooltipGroup,
} from "@/components/tooltip/CustomTooltip";
import { useRef, useState } from "react";
import { copyAll, handleCopy } from "../copyingUtils";
import RegenerateOption from "../../components/RegenerateOption";
import { FiChevronDown, FiChevronUp, FiTool } from "react-icons/fi";

// Multi-tool renderer component for grouped tools
function MultiToolRenderer({
  packets,
  chatState,
}: {
  packets: Packet[];
  chatState: FullChatState;
}) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Group packets by tool instance (consecutive tool start/delta/end sequences)
  const toolGroups: Packet[][] = [];
  let currentGroup: Packet[] = [];

  for (const packet of packets) {
    if (packet.obj.type === PacketType.TOOL_START) {
      // Start a new group
      if (currentGroup.length > 0) {
        toolGroups.push(currentGroup);
      }
      currentGroup = [packet];
    } else {
      // Add to current group
      currentGroup.push(packet);
    }
  }

  // Don't forget the last group
  if (currentGroup.length > 0) {
    toolGroups.push(currentGroup);
  }

  const toggleExpanded = () => setIsExpanded(!isExpanded);

  // Get summary information
  const totalTools = toolGroups.length;
  const toolTypes = new Set(
    toolGroups.map((group) => {
      const startPacket = group.find(
        (p) => p.obj.type === PacketType.TOOL_START
      );
      return startPacket ? (startPacket.obj as any).tool_name : "unknown";
    })
  );

  // Check if any tool is still in progress
  const hasActiveTools = toolGroups.some((group) => {
    const hasStart = group.some((p) => p.obj.type === PacketType.TOOL_START);
    const hasEnd = group.some((p) => p.obj.type === PacketType.TOOL_END);
    return hasStart && !hasEnd;
  });

  return (
    <div className="mb-3">
      {/* Header */}
      <div
        className="flex items-center justify-between py-2 px-3 border border-gray-200 dark:border-gray-700 rounded-t cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
        onClick={toggleExpanded}
      >
        <div className="flex items-center gap-2">
          <FiTool className="w-3 h-3 text-gray-600 dark:text-gray-400" />
          <div>
            <h3 className="text-xs font-medium text-gray-700 dark:text-gray-300">
              {totalTools} tool{totalTools !== 1 ? "s" : ""} used
              {toolTypes.size > 1 && (
                <span className="text-gray-500">
                  {" "}
                  • {Array.from(toolTypes).join(", ")}
                </span>
              )}
              {hasActiveTools && (
                <span className="ml-2 inline-flex items-center">
                  <div className="w-1 h-1 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="ml-1 text-blue-600 dark:text-blue-400">
                    Active
                  </span>
                </span>
              )}
            </h3>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500 dark:text-gray-500">
            {isExpanded ? "Hide" : "Show"}
          </span>
          {isExpanded ? (
            <FiChevronUp className="w-3 h-3 text-gray-500" />
          ) : (
            <FiChevronDown className="w-3 h-3 text-gray-500" />
          )}
        </div>
      </div>

      {/* Expandable content */}
      {isExpanded && (
        <div className="border-l border-r border-b border-gray-200 dark:border-gray-700 rounded-b bg-white dark:bg-gray-900">
          <div className="p-4">
            {toolGroups.map((toolGroup, index) => (
              <div key={index} className="relative">
                {/* Connecting line (except for the last item) */}
                {index < toolGroups.length - 1 && (
                  <div className="absolute left-4 top-full w-px h-4 bg-gray-300 dark:bg-gray-600 z-10"></div>
                )}

                {/* Tool content */}
                <div className="relative">
                  {/* Tool indicator dot */}
                  <div className="absolute -left-1 top-3 w-2 h-2 bg-blue-500 rounded-full border-2 border-white dark:border-gray-900 z-20"></div>

                  {/* Tool content with left margin for the connection line */}
                  <div className="ml-6">
                    {renderMessageComponent({ packets: toolGroup }, chatState)}
                  </div>
                </div>

                {/* Spacing between tools */}
                {index < toolGroups.length - 1 && <div className="h-4"></div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function SimpleMessage({
  rawPackets,
  chatState,
}: {
  rawPackets: Packet[];
  chatState: FullChatState;
}) {
  const markdownRef = useRef<HTMLDivElement>(null);
  const [isRegenerateDropdownVisible, setIsRegenerateDropdownVisible] =
    useState(false);

  // Group all chat packets together by ind
  const groupedChatPacketsByInd: Map<number, Packet[]> = rawPackets.reduce(
    (acc: Map<number, Packet[]>, packet) => {
      const ind = packet.ind;
      if (!acc.has(ind)) {
        acc.set(ind, []);
      }
      acc.get(ind)!.push(packet);
      return acc;
    },
    new Map()
  );

  // Helper function to check if packets are tool packets
  const isToolGroup = (packets: Packet[]) => {
    return packets.some(
      (packet) =>
        packet.obj.type === PacketType.TOOL_START ||
        packet.obj.type === PacketType.TOOL_DELTA ||
        packet.obj.type === PacketType.TOOL_END
    );
  };

  // Create smart groups that combine consecutive tool groups
  const createSmartGroups = () => {
    const sortedEntries = Array.from(groupedChatPacketsByInd.entries()).sort(
      ([a], [b]) => a - b
    );
    const smartGroups: {
      packets: Packet[];
      isMultiTool: boolean;
      inds: number[];
    }[] = [];
    let currentToolGroup: { packets: Packet[]; inds: number[] } | null = null;

    for (const [ind, packets] of sortedEntries) {
      const isThisGroupTools = isToolGroup(packets);

      if (isThisGroupTools) {
        if (currentToolGroup) {
          // Add to existing tool group
          currentToolGroup.packets.push(...packets);
          currentToolGroup.inds.push(ind);
        } else {
          // Start new tool group
          currentToolGroup = { packets: [...packets], inds: [ind] };
        }
      } else {
        // Non-tool group - finalize any pending tool group first
        if (currentToolGroup) {
          smartGroups.push({
            packets: currentToolGroup.packets,
            isMultiTool: currentToolGroup.inds.length > 1,
            inds: currentToolGroup.inds,
          });
          currentToolGroup = null;
        }
        // Add the non-tool group
        smartGroups.push({
          packets,
          isMultiTool: false,
          inds: [ind],
        });
      }
    }

    // Don't forget any pending tool group
    if (currentToolGroup) {
      smartGroups.push({
        packets: currentToolGroup.packets,
        isMultiTool: currentToolGroup.inds.length > 1,
        inds: currentToolGroup.inds,
      });
    }

    return smartGroups;
  };

  const smartGroups = createSmartGroups();

  // Check if streaming is complete (has STOP packet)
  const isStreamingComplete = rawPackets.some(
    (packet) => packet.obj.type === PacketType.STOP
  );

  // Extract text content for copying
  const getTextContent = () => {
    return smartGroups
      .map((group) => {
        // Extract text from packets - this is a simplified approach
        return group.packets
          .map((packet) => {
            if (
              packet.obj.type === PacketType.MESSAGE_START ||
              packet.obj.type === PacketType.MESSAGE_DELTA
            ) {
              return (packet.obj as MessageStart | MessageDelta).content || "";
            }
            return "";
          })
          .join("")
          .trim();
      })
      .join("\n")
      .trim();
  };

  // Return a list of rendered message components, one for each ind
  return (
    <div className="py-5 ml-4 lg:px-5 relative flex">
      <div className="mx-auto w-[90%] max-w-message-max">
        <div className="lg:mr-12 mobile:ml-0 md:ml-8">
          <div className="flex items-start">
            <AssistantIcon
              className="mobile:hidden"
              size={24}
              assistant={chatState.assistant}
            />
            <div className="w-full">
              <div className="max-w-message-max break-words">
                <div className="w-full desktop:ml-4">
                  <div className="max-w-message-max break-words">
                    <div
                      ref={markdownRef}
                      className="overflow-x-visible max-w-content-max focus:outline-none cursor-text select-text"
                      onCopy={(e) => handleCopy(e, markdownRef)}
                    >
                      {smartGroups.map((group, index) => (
                        <div key={group.inds.join("-")}>
                          {group.isMultiTool ? (
                            <MultiToolRenderer
                              packets={group.packets}
                              chatState={chatState}
                            />
                          ) : (
                            renderMessageComponent(
                              { packets: group.packets },
                              chatState
                            )
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Feedback buttons - only show when streaming is complete */}
                  {chatState.handleFeedback && isStreamingComplete && (
                    <div className="flex md:flex-row gap-x-0.5 mt-1 transition-transform duration-300 ease-in-out transform opacity-100">
                      <TooltipGroup>
                        <div className="flex justify-start w-full gap-x-0.5">
                          <CustomTooltip showTick line content="Copy">
                            <CopyButton
                              copyAllFn={() =>
                                copyAll(getTextContent(), markdownRef)
                              }
                            />
                          </CustomTooltip>
                          <CustomTooltip showTick line content="Good response">
                            <HoverableIcon
                              icon={<LikeFeedback />}
                              onClick={() => chatState.handleFeedback("like")}
                            />
                          </CustomTooltip>
                          <CustomTooltip showTick line content="Bad response">
                            <HoverableIcon
                              icon={<DislikeFeedback size={16} />}
                              onClick={() =>
                                chatState.handleFeedback("dislike")
                              }
                            />
                          </CustomTooltip>
                          {chatState.regenerate && (
                            <CustomTooltip
                              disabled={isRegenerateDropdownVisible}
                              showTick
                              line
                              content="Regenerate"
                            >
                              <RegenerateOption
                                onDropdownVisibleChange={
                                  setIsRegenerateDropdownVisible
                                }
                                selectedAssistant={chatState.assistant}
                                regenerate={chatState.regenerate}
                                overriddenModel={chatState.overriddenModel}
                              />
                            </CustomTooltip>
                          )}
                        </div>
                      </TooltipGroup>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
