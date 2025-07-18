import React, { useState } from "react";
import {
  FiSearch,
  FiChevronDown,
  FiChevronUp,
  FiFile,
  FiExternalLink,
  FiClock,
} from "react-icons/fi";
import {
  ChatPacket,
  PacketType,
  SearchToolPacket,
  SearchToolStart,
  SearchToolEnd,
} from "../../services/streamingModels";
import { MessageRenderer } from "./interfaces";
import { SearchResultIcon } from "@/components/SearchResultIcon";

export const SearchToolRenderer: MessageRenderer<SearchToolPacket, {}> = ({
  packets,
}: {
  packets: SearchToolPacket[];
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const searchStart = packets.find(
    (packet) => packet.obj.type === PacketType.SEARCH_TOOL_START
  )?.obj as SearchToolStart;

  const searchEnd = packets.find(
    (packet) => packet.obj.type === PacketType.SEARCH_TOOL_END
  )?.obj as SearchToolEnd;

  const query = searchStart?.query;
  const results = searchEnd?.results || [];
  const isSearching = searchStart && !searchEnd;
  const isComplete = searchStart && searchEnd;

  const toggleExpanded = () => setIsExpanded(!isExpanded);

  // Loading state - when searching
  if (isSearching) {
    return (
      <div className="flex items-center gap-2 py-2 px-3 border border-gray-200 dark:border-gray-700 rounded mb-3">
        <FiSearch className="w-3 h-3 text-gray-600 dark:text-gray-400 animate-pulse" />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
              Searching
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
          {query && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              "{query}"
            </p>
          )}
        </div>
      </div>
    );
  }

  // Complete state - show results
  if (isComplete) {
    return (
      <div className="mb-3">
        {/* Header */}
        <div
          className="flex items-center justify-between py-2 px-3 border border-gray-200 dark:border-gray-700 rounded-t cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          onClick={toggleExpanded}
        >
          <div className="flex items-center gap-2">
            <FiSearch className="w-3 h-3 text-gray-600 dark:text-gray-400" />
            <div>
              <h3 className="text-xs font-medium text-gray-700 dark:text-gray-300">
                Search Results
              </h3>
              {query && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  "{query}" • {results.length} document
                  {results.length !== 1 ? "s" : ""} found
                </p>
              )}
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
            {results.length > 0 ? (
              <div className="p-3 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {results.map((doc, index) => (
                  <div
                    key={doc.document_id || index}
                    className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500 transition-all group"
                  >
                    <div className="flex items-start gap-2">
                      {/* Icon */}
                      <div className="flex-shrink-0 mt-0.5">
                        {doc.link ? (
                          <SearchResultIcon url={doc.link} />
                        ) : (
                          <div className="w-4 h-4 flex items-center justify-center bg-gray-100 dark:bg-gray-700 rounded">
                            <FiFile className="w-2.5 h-2.5 text-gray-500" />
                          </div>
                        )}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-1">
                          <h4 className="font-medium text-xs line-clamp-2 text-gray-800 dark:text-gray-200 group-hover:text-black dark:group-hover:text-white transition-colors">
                            {doc.semantic_identifier || "Untitled Document"}
                          </h4>
                          {doc.link && (
                            <a
                              href={doc.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <FiExternalLink className="w-2.5 h-2.5 text-gray-500 hover:text-gray-700" />
                            </a>
                          )}
                        </div>

                        {/* Source type badge */}
                        <div className="flex items-center gap-1 mt-1 mb-1">
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                            {doc.source_type}
                          </span>
                          {doc.updated_at && (
                            <span className="flex items-center gap-0.5 text-xs text-gray-500">
                              <FiClock className="w-2.5 h-2.5" />
                              {new Date(doc.updated_at).toLocaleDateString()}
                            </span>
                          )}
                        </div>

                        {/* Snippet */}
                        {doc.blurb && (
                          <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 leading-relaxed">
                            {doc.blurb}
                          </p>
                        )}

                        {/* Highlights */}
                        {doc.match_highlights &&
                          doc.match_highlights.length > 0 && (
                            <div className="mt-1 pt-1 border-t border-gray-200 dark:border-gray-700">
                              <p className="text-xs text-gray-500 mb-0.5">
                                Highlights:
                              </p>
                              <div className="space-y-0.5">
                                {doc.match_highlights
                                  .slice(0, 1)
                                  .map((highlight, idx) => (
                                    <p
                                      key={idx}
                                      className="text-xs text-gray-600 dark:text-gray-400 line-clamp-1"
                                      dangerouslySetInnerHTML={{
                                        __html: highlight.replace(
                                          /<hi>(.*?)<\/hi>/g,
                                          '<span class="bg-gray-200 dark:bg-gray-700 px-0.5 rounded font-medium">$1</span>'
                                        ),
                                      }}
                                    />
                                  ))}
                              </div>
                            </div>
                          )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                <FiSearch className="w-6 h-6 mx-auto mb-1 opacity-50" />
                <p className="text-xs">No documents found</p>
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
