import { ValidSources } from "@/lib/types";

// Base interface for all streaming objects
interface BaseObj {
  type: string;
}

export enum PacketType {
  MESSAGE_START = "message_start",
  MESSAGE_DELTA = "message_delta",
  MESSAGE_END = "message_end",

  STOP = "stop",

  SEARCH_TOOL_START = "search_tool_start",
  SEARCH_TOOL_END = "search_tool_end",
  IMAGE_TOOL_START = "image_tool_start",
  IMAGE_TOOL_END = "image_tool_end",
}

// LlmDoc interface matching the backend model
export interface LlmDoc {
  document_id: string;
  content: string;
  blurb: string;
  semantic_identifier: string;
  source_type: ValidSources;
  metadata: { [key: string]: string | string[] };
  updated_at: string | null;
  link: string | null;
  source_links: { [key: number]: string } | null;
  match_highlights: string[] | null;
}

// Streaming message objects
export interface MessageStart extends BaseObj {
  id: string;
  type: PacketType.MESSAGE_START;
  content: string;
}

export interface MessageDelta extends BaseObj {
  content: string;
  type: PacketType.MESSAGE_DELTA;
}

export interface MessageEnd extends BaseObj {
  type: PacketType.MESSAGE_END;
}

export interface Stop extends BaseObj {
  type: PacketType.STOP;
}

export type ChatObj = MessageStart | MessageDelta | MessageEnd;

export type StopObj = Stop;

export interface SearchToolStart extends BaseObj {
  type: PacketType.SEARCH_TOOL_START;
  query: string;
}

export interface SearchToolEnd extends BaseObj {
  type: PacketType.SEARCH_TOOL_END;
  results: LlmDoc[];
}

export type SearchToolObj = SearchToolStart | SearchToolEnd;

export interface ImageToolStart extends BaseObj {
  type: PacketType.IMAGE_TOOL_START;
  prompt: string;
}

export interface ImageToolEnd extends BaseObj {
  type: PacketType.IMAGE_TOOL_END;
  images: Array<{
    id: string;
    url: string;
    prompt: string;
  }>;
}

export type ImageToolObj = ImageToolStart | ImageToolEnd;

// Union type for all possible streaming objects
export type ObjTypes = ChatObj | SearchToolObj | ImageToolObj | StopObj;

// Packet wrapper for streaming objects
export interface Packet {
  ind: number;
  obj: ObjTypes;
}

export interface ChatPacket {
  ind: number;
  obj: ChatObj;
}

export interface SearchToolPacket {
  ind: number;
  obj: SearchToolObj;
}

export interface ImageToolPacket {
  ind: number;
  obj: ImageToolObj;
}

export interface StopPacket {
  ind: number;
  obj: StopObj;
}
