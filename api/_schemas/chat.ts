/**
 * Zod validation schemas for AI SDK messages
 * Based on UIMessage type from @ai-sdk/react
 * 
 * @see https://sdk.vercel.ai/docs/reference/ai-sdk-ui/use-chat
 */

import { z } from "zod";

/**
 * Text part of a UI message
 */
const textPartSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
  state: z.enum(["streaming", "done"]).optional(),
  providerMetadata: z.record(z.unknown()).optional(),
});

/**
 * File part of a UI message
 */
const filePartSchema = z.object({
  type: z.literal("file"),
  mediaType: z.string(),
  filename: z.string().optional(),
  url: z.string().url(),
  providerMetadata: z.record(z.unknown()).optional(),
});

/**
 * Tool invocation part (pending execution)
 */
const toolInvocationPartSchema = z.object({
  type: z.literal("tool-invocation"),
  toolCallId: z.string(),
  toolName: z.string(),
  args: z.record(z.unknown()),
  state: z.enum(["input-streaming", "input-available"]).optional(),
});

/**
 * Dynamic tool part (with result)
 */
const dynamicToolPartSchema = z.object({
  type: z.literal("dynamic-tool"),
  toolName: z.string(),
  toolCallId: z.string(),
  title: z.string().optional(),
  providerExecuted: z.boolean().optional(),
  state: z.enum([
    "input-streaming",
    "input-available",
    "approval-requested",
    "approval-responded",
    "output-available",
    "output-error",
    "output-denied",
  ]),
  input: z.unknown().optional(),
  output: z.unknown().optional(),
  errorText: z.string().optional(),
  callProviderMetadata: z.record(z.unknown()).optional(),
  preliminary: z.boolean().optional(),
  approval: z
    .object({
      id: z.string(),
      approved: z.boolean().optional(),
      reason: z.string().optional(),
    })
    .optional(),
});

/**
 * Generic tool part for any other tool types not explicitly defined
 * This catches custom tool types like "tool-displayRestaurants"
 */
const genericToolPartSchema = z.object({
  type: z.string(), // Allow any string for type
  state: z.string().optional(),
  // Allow any other properties
}).passthrough();

/**
 * Union of all possible message part types
 */
const messagePartSchema = z.discriminatedUnion("type", [
  textPartSchema,
  filePartSchema,
  toolInvocationPartSchema,
  dynamicToolPartSchema,
]).or(genericToolPartSchema); // Fallback to generic tool schema

/**
 * UI Message schema matching AI SDK's UIMessage type
 */
const uiMessageSchema = z.object({
  id: z.string(),
  role: z.enum(["system", "user", "assistant"]),
  content: z.string().optional(), // Legacy: some messages may have content string
  parts: z.array(messagePartSchema).optional(), // New: parts-based messages
  metadata: z.record(z.unknown()).optional(),
  createdAt: z.coerce.date().optional(),
});

/**
 * Chat request body schema
 */
export const chatRequestSchema = z.object({
  messages: z
    .array(uiMessageSchema)
    .min(1, "At least one message is required")
    .refine(
      (messages) => {
        // At least one message must be from the user
        return messages.some((msg) => msg.role === "user");
      },
      {
        message: "At least one user message is required",
      }
    ),
  // Optional fields that might be sent by the client
  context: z.record(z.unknown()).optional(),
  data: z.record(z.unknown()).optional(),
});

/**
 * Type inference for validated chat request
 */
export type ChatRequest = z.infer<typeof chatRequestSchema>;

/**
 * Type inference for validated UI message
 */
export type ValidatedUIMessage = z.infer<typeof uiMessageSchema>;

/**
 * Helper function to validate and parse chat request
 * Returns parsed data or throws ZodError with detailed error messages
 */
export function validateChatRequest(body: unknown): ChatRequest {
  return chatRequestSchema.parse(body);
}

/**
 * Safe validation that returns success/error result
 */
export function safeParseChatRequest(body: unknown) {
  return chatRequestSchema.safeParse(body);
}

