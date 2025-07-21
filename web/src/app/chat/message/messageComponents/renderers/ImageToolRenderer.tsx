import React, { useState } from "react";
import {
  FiImage,
  FiChevronDown,
  FiChevronUp,
  FiDownload,
  FiEye,
} from "react-icons/fi";
import {
  PacketType,
  ToolPacket,
  ToolStart,
  ToolEnd,
  ToolDelta,
} from "../../../services/streamingModels";
import { MessageRenderer } from "../interfaces";
import { buildImgUrl } from "../../../components/files/images/utils";

export const ImageToolRenderer: MessageRenderer<ToolPacket, {}> = ({
  packets,
}: {
  packets: ToolPacket[];
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const imageStart = packets.find(
    (packet) => packet.obj.type === PacketType.TOOL_START
  )?.obj as ToolStart;

  const imageDeltas = packets
    .filter((packet) => packet.obj.type === PacketType.TOOL_DELTA)
    .map((packet) => packet.obj as ToolDelta);

  const imageEnd = packets.find(
    (packet) => packet.obj.type === PacketType.TOOL_END
  )?.obj as ToolEnd;

  const prompt = imageStart?.tool_main_description;
  const images = imageDeltas.flatMap((delta) => delta?.images || []);
  const isGenerating = imageStart && !imageEnd;
  const isComplete = imageStart && imageEnd;

  const toggleExpanded = () => setIsExpanded(!isExpanded);

  // Loading state - when generating
  if (isGenerating) {
    return (
      <div className="flex items-center gap-2 py-2 px-3 border border-gray-200 dark:border-gray-700 rounded mb-3">
        <FiImage className="w-3 h-3 text-gray-600 dark:text-gray-400 animate-pulse" />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
              Generating images{prompt && ` "${prompt}"`}
            </span>
            <div className="flex gap-0.5">
              <div className="w-0.5 h-0.5 bg-gray-500 rounded-full animate-bounce"></div>
              <div
                className="w-0.5 h-0.5 bg-gray-500 rounded-full animate-bounce"
                style={{ animationDelay: "0.1s" }}
              ></div>
              <div
                className="w-0.5 h-0.5 bg-gray-500 rounded-full animate-bounce"
                style={{ animationDelay: "0.2s" }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Complete state - show images
  if (isComplete) {
    return (
      <div className="mb-3">
        {/* Header */}
        <div
          className="flex items-center justify-between py-2 px-3 border border-gray-200 dark:border-gray-700 rounded-t cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          onClick={toggleExpanded}
        >
          <div className="flex items-center gap-2">
            <FiImage className="w-3 h-3 text-gray-600 dark:text-gray-400" />
            <div>
              <h3 className="text-xs font-medium text-gray-700 dark:text-gray-300">
                {prompt && ` "${prompt}"`} • {images.length} image
                {images.length !== 1 ? "s" : ""} generated
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
            {images.length > 0 ? (
              <div className="p-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                {images.map(
                  (image: { [key: string]: string }, index: number) => (
                    <div
                      key={image.id || index}
                      className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500 transition-all group"
                    >
                      {/* Image */}
                      {image.id && (
                        <div className="relative mb-2">
                          <img
                            src={buildImgUrl(image.id)}
                            alt={image.prompt || "Generated image"}
                            className="w-full h-48 object-cover rounded"
                            loading="lazy"
                          />
                          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                            <button
                              onClick={() =>
                                window.open(buildImgUrl(image.id!), "_blank")
                              }
                              className="bg-black bg-opacity-50 text-white p-1 rounded hover:bg-opacity-70 transition-colors"
                              title="View full size"
                            >
                              <FiEye className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => {
                                const link = document.createElement("a");
                                link.href = buildImgUrl(image.id!);
                                link.download = `generated-image-${index + 1}.png`;
                                link.click();
                              }}
                              className="bg-black bg-opacity-50 text-white p-1 rounded hover:bg-opacity-70 transition-colors"
                              title="Download"
                            >
                              <FiDownload className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Prompt */}
                      {image.prompt && (
                        <div className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 leading-relaxed">
                          <span className="font-medium">Prompt:</span>{" "}
                          {image.prompt}
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
            ) : (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                <FiImage className="w-6 h-6 mx-auto mb-1 opacity-50" />
                <p className="text-xs">No images generated</p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // Fallback (shouldn't happen in normal flow)
  return <div></div>;
};
