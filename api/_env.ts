/**
 * Environment variables configuration
 * Provides type-safe access to environment variables
 */

import { config } from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

// Load .env.local FIRST, before validation
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
config({ path: path.resolve(__dirname, "../.env.local") });

// Simple environment variable access with defaults
export const env = {
  // Google Gemini API configuration
  GOOGLE_API_KEY: process.env.GOOGLE_API_KEY,
  GOOGLE_GENERATIVE_AI_API_KEY: process.env.GOOGLE_GENERATIVE_AI_API_KEY,

  // MCP Server configuration
  MCP_SERVER_URL: process.env.MCP_SERVER_URL || "",
  MCP_API_KEY: process.env.MCP_API_KEY || "",
  MCP_ANALYSIS_ID: process.env.MCP_ANALYSIS_ID || "",

  // API server configuration
  API_PORT: parseInt(process.env.API_PORT || "3000", 10),
  NODE_ENV: (process.env.NODE_ENV || "development") as
    | "development"
    | "production"
    | "test",
};

/**
 * Helper to get Google API key with fallback
 * Google SDKs accept either GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY
 */
export const getGoogleApiKey = (): string => {
  const key = env.GOOGLE_API_KEY || env.GOOGLE_GENERATIVE_AI_API_KEY;
  if (!key) {
    throw new Error(
      "❌ Missing Google API key. Set either GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY"
    );
  }
  return key;
};
