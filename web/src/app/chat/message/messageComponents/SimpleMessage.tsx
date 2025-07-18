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

  // Check if streaming is complete (has STOP packet)
  const isStreamingComplete = rawPackets.some(
    (packet) => packet.obj.type === PacketType.STOP
  );

  // Extract text content for copying
  const getTextContent = () => {
    return Array.from(groupedChatPacketsByInd.entries())
      .map(([ind, packets]) => {
        // Extract text from packets - this is a simplified approach
        return packets
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
                      {Array.from(groupedChatPacketsByInd.entries()).map(
                        ([ind, packets]) => (
                          <div key={ind}>
                            {renderMessageComponent(
                              { packets: packets },
                              chatState
                            )}
                          </div>
                        )
                      )}
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
