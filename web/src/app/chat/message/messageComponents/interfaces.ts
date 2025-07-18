import { MinimalPersonaSnapshot } from "@/app/admin/assistants/interfaces";
import { FeedbackType } from "../../interfaces";
import { Packet } from "../../services/streamingModels";
import { OnyxDocument, MinimalOnyxDocument } from "@/lib/search/interfaces";
import { FileResponse } from "../../my-documents/DocumentsContext";
import { LlmDescriptor } from "@/lib/hooks";

export interface FullChatState {
  handleFeedback: (feedback: FeedbackType) => void;
  assistant: MinimalPersonaSnapshot;
  // Document-related context for citations
  docs?: OnyxDocument[] | null;
  userFiles?: FileResponse[];
  citations?: { [key: string]: number };
  setPresentingDocument?: (document: MinimalOnyxDocument) => void;
  // Regenerate functionality
  regenerate?: (modelOverRide: LlmDescriptor) => Promise<void>;
  overriddenModel?: string;
}

interface MessageRendererProps<
  T extends Packet,
  S extends Partial<FullChatState>,
> {
  packets: T[];
  state: S;
}

export type MessageRenderer<
  T extends Packet,
  S extends Partial<FullChatState>,
> = ({ packets, state }: MessageRendererProps<T, S>) => JSX.Element;
