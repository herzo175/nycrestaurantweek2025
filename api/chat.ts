import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import {
  streamText,
  stepCountIs,
  convertToModelMessages,
  tool as createTool,
  type ToolSet,
  type UIMessage,
} from "ai";
import { z } from "zod";
import { experimental_createMCPClient as createMCPClient } from "@ai-sdk/mcp";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import type { Restaurant } from "../src/types/restaurant.js";
import { GeometryCache } from "./_utils/geometryOptimizer.js";
import { point, booleanPointInPolygon } from "@turf/turf";
import type { Feature, Polygon, MultiPolygon } from "geojson";
import { wrapToolsWithGeometryOptimization } from "./_utils/toolWrapper.js";
import { env, getGoogleApiKey } from "./_env.js";
import { safeParseChatRequest } from "./_schemas/chat.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load restaurant data for local search
const restaurantsPath = path.join(__dirname, "../src/data/FinalData.json");
let allRestaurants: Restaurant[] = [];
try {
  const restaurantsData = fs.readFileSync(restaurantsPath, "utf8");
  allRestaurants = JSON.parse(restaurantsData);
  console.log(
    `📍 Loaded ${allRestaurants.length} restaurants from FinalData.json`
  );
} catch (error) {
  console.error("❌ Failed to load restaurant data:", error);
}

// ============ FUZZY MATCHING HELPERS ============

/**
 * Normalize string for fuzzy matching
 */
function normalizeForMatching(str: string): string {
  return str
    .toLowerCase()
    .replace(/^(the|a|an)\s+/i, "") // Remove leading articles
    .replace(/\s+and\s+/g, " ") // Remove "and" between words
    .replace(/[^a-z0-9\s]/g, "") // Remove special chars
    .replace(/\s+/g, " ") // Collapse multiple spaces
    .trim();
}

/**
 * Calculate Levenshtein distance between two strings
 */
function levenshteinDistance(str1: string, str2: string): number {
  const len1 = str1.length;
  const len2 = str2.length;
  const matrix: number[][] = Array(len1 + 1)
    .fill(null)
    .map(() => Array(len2 + 1).fill(0));

  for (let i = 0; i <= len1; i++) matrix[i][0] = i;
  for (let j = 0; j <= len2; j++) matrix[0][j] = j;

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }
  return matrix[len1][len2];
}

/**
 * Fuzzy match a restaurant by name or slug
 * Returns the best match from the search pool
 */
function fuzzyMatchRestaurant(
  input: string,
  searchPool: Restaurant[]
): Restaurant | null {
  if (!input) return null;

  const normalizedInput = normalizeForMatching(input);
  const inputSlug = input.toLowerCase().replace(/\s+/g, "-");

  // Tier 1: Exact slug match
  const exactMatch = searchPool.find(
    (r) => r.slug === inputSlug || r.slug === input.toLowerCase()
  );
  if (exactMatch) {
    console.log(`✅ Exact slug match: "${input}" → "${exactMatch.name}"`);
    return exactMatch;
  }

  // Tier 2: Normalized name match
  const normalizedMatch = searchPool.find(
    (r) => normalizeForMatching(r.name) === normalizedInput
  );
  if (normalizedMatch) {
    console.log(
      `✅ Normalized name match: "${input}" → "${normalizedMatch.name}"`
    );
    return normalizedMatch;
  }

  // Tier 3: Partial name match
  const partialMatch = searchPool.find((r) => {
    const normalizedName = normalizeForMatching(r.name);
    return (
      normalizedName.includes(normalizedInput) ||
      normalizedInput.includes(normalizedName)
    );
  });
  if (partialMatch) {
    console.log(`✅ Partial name match: "${input}" → "${partialMatch.name}"`);
    return partialMatch;
  }

  // Tier 4: Slug similarity match
  const slugMatch = searchPool.find((r) =>
    r.slug.includes(normalizedInput.replace(/\s+/g, "-"))
  );
  if (slugMatch) {
    console.log(`✅ Slug similarity match: "${input}" → "${slugMatch.name}"`);
    return slugMatch;
  }

  // Tier 5: Levenshtein distance match (typos)
  const threshold = normalizedInput.length < 8 ? 2 : 3;
  const typoMatch = searchPool.find((r) => {
    const normalizedName = normalizeForMatching(r.name);
    return levenshteinDistance(normalizedInput, normalizedName) <= threshold;
  });
  if (typoMatch) {
    console.log(
      `✅ Typo match (distance ≤${threshold}): "${input}" → "${typoMatch.name}"`
    );
    return typoMatch;
  }

  console.log(`❌ No fuzzy match found for: "${input}"`);
  return null;
}

// Type definitions for tool results
interface ToolResult {
  isError?: boolean;
  [key: string]: unknown;
}

interface SearchDocumentsResult extends ToolResult {
  chunks?: Array<{
    text?: string;
    [key: string]: unknown;
  }>;
}

interface ExecuteSqlResult extends ToolResult {
  [key: string]: unknown;
}

interface GetIsolineResult extends ToolResult {
  geojson?: Feature<Polygon | MultiPolygon> | Polygon | MultiPolygon;
  geometry?: Feature<Polygon | MultiPolygon> | Polygon | MultiPolygon;
  results?: Array<{
    geojson?: Feature<Polygon | MultiPolygon> | Polygon | MultiPolygon;
    [key: string]: unknown;
  }>;
  restaurants?: Array<{
    name: string;
    slug: string;
    cuisine?: string;
    price?: string;
    neighborhood?: string;
    yelp_rating?: number;
    michelin_award?: string;
    nyttop100_rank?: string;
  }>;
  restaurantSlugs?: string[];
  count?: number;
  filterPoolActive?: boolean;
  filterPoolSize?: number;
  searchedPool?: number;
}

const app = new Hono();

// Enable CORS (safe to have in mounted app, also works for local dev)
app.use("/*", cors());

// Initialize Google Gemini with validated API key
const google = createGoogleGenerativeAI({
  apiKey: getGoogleApiKey(),
});

/**
 * POST /chat
 *
 * Streaming chat endpoint using AI SDK + Google Gemini + MCP
 * Returns AI SDK stream format compatible with useChat hook
 */

// Handler function - shared for both route patterns (Vercel and local)
const chatHandler = async (c: Context) => {
  console.log("✅ [chat.ts] Chat handler invoked!");
  console.log("   Path:", c.req.path);
  console.log("   URL:", c.req.url);

  // Validate request body with Zod
  const body = await c.req.json();
  const parseResult = safeParseChatRequest(body);

  if (!parseResult.success) {
    const errors = parseResult.error.format();
    console.error("❌ Invalid chat request:", JSON.stringify(errors, null, 2));

    // Log the actual messages being sent for debugging
    if (body.messages) {
      console.error(
        "📨 Received messages:",
        JSON.stringify(body.messages, null, 2)
      );
    }

    return c.json(
      {
        error: "Invalid request body",
        details: errors,
        message: "Please check that messages array is properly formatted",
      },
      400
    );
  }

  const { messages: rawMessages, context } = parseResult.data;
  console.log(`📨 Received ${rawMessages.length} messages from client`);

  // Extract filterPool from context (restaurants matching current filter bar selections)
  type ChatContext = {
    filterPool?: string[];
    [key: string]: unknown;
  };
  const chatContext = context as ChatContext | undefined;
  const filterPool: string[] = chatContext?.filterPool || [];
  const hasFilterPool =
    filterPool.length > 0 && filterPool.length < allRestaurants.length;
  if (hasFilterPool) {
    console.log(
      `🎯 Filter pool active: ${filterPool.length} restaurants (of ${allRestaurants.length} total)`
    );
  } else {
    console.log(
      `🎯 No filter pool - searching all ${allRestaurants.length} restaurants`
    );
  }

  // Normalize messages: ensure parts is always an array (AI SDK requirement)
  // Type assertion is safe here because we've validated the structure with Zod
  const messages = rawMessages.map((msg) => ({
    ...msg,
    parts: msg.parts ?? [], // Convert undefined to empty array
  })) as UIMessage[];

  const analysisId = env.MCP_ANALYSIS_ID;
  let mcpTools: ToolSet = {};
  let datasetContext = "";
  let mcpClient: Awaited<ReturnType<typeof createMCPClient>>;
  let sqlTables = "";
  let docCollections = "";
  let spatialReference = "";

  // Initialize geometry cache for this request
  const geometryCache = new GeometryCache();

  // 1. Connect and Fetch Tools/Resources (Let it fail if server is down)
  const transport = new StreamableHTTPClientTransport(
    new URL(env.MCP_SERVER_URL),
    {
      requestInit: {
        headers: { Authorization: `Bearer ${env.MCP_API_KEY}` },
      },
    }
  );

  mcpClient = await createMCPClient({ transport });
  console.log("✅ Connected to MCP");

  // Fetch Spatial Reference and Examples
  try {
    console.log("📂 Fetching spatial reference and examples...");
    const [spatialFuncs, spatialExamples] = await Promise.all([
      mcpClient.readResource({ uri: "spatial-functions://reference" }),
      mcpClient.readResource({ uri: "spatial-query-examples://duckdb" }),
    ]);

    if (spatialFuncs?.contents?.[0]?.text) {
      spatialReference += `\n### DUCKDB SPATIAL FUNCTIONS REFERENCE:\n${spatialFuncs.contents[0].text}\n`;
    }
    if (spatialExamples?.contents?.[0]?.text) {
      spatialReference += `\n### SPATIAL QUERY EXAMPLES:\n${spatialExamples.contents[0].text}\n`;
    }
  } catch (err) {
    console.error("⚠️ Failed to fetch spatial resources:", err);
  }

  mcpTools = await mcpClient.tools();
  console.log("📦 MCP Tools:", Object.keys(mcpTools).join(", "));

  // Wrap MCP tools with geometry optimization
  const optimizedTools = wrapToolsWithGeometryOptimization(
    mcpTools,
    geometryCache
  );

  // Create a custom tool for displaying restaurant cards
  const displayRestaurantsTool = {
    description:
      "Render interactive restaurant cards in the chat UI. Use this after finding restaurants via search_documents/execute_sql to display results.",
    inputSchema: z.object({
      restaurant_names: z
        .array(z.string())
        .describe("List of restaurant names or slugs to display as cards."),
      query: z
        .string()
        .optional()
        .describe("The original search query (e.g., 'Italian restaurants')."),
    }),
    execute: async (params: { restaurant_names: string[]; query?: string }) => {
      const { restaurant_names, query } = params;
      try {
        console.log(
          `🍽️ displayRestaurants resolving cards for: ${JSON.stringify(
            restaurant_names
          )}`
        );

        // Start with filterPool-restricted set if active, otherwise all restaurants
        const searchPool = hasFilterPool
          ? allRestaurants.filter((r) => filterPool.includes(r.slug))
          : allRestaurants;

        // Match by slug or exact name
        const foundRestaurants = searchPool.filter(
          (r) =>
            restaurant_names.includes(r.slug) ||
            restaurant_names.some(
              (name) => r.name.toLowerCase() === name.toLowerCase()
            )
        );

        // Sort by yelp_rating (highest first) for best recommendations
        foundRestaurants.sort((a, b) => {
          const aRating = a.yelp_rating || 0;
          const bRating = b.yelp_rating || 0;
          return bRating - aRating;
        });

        console.log(`✅ Resolved ${foundRestaurants.length} restaurant cards`);

        // Return results with clean data structure (all fields needed for RestaurantCard)
        // Limit to 5 cards - sorted by yelp_rating (highest first)
        const restaurants: Restaurant[] = foundRestaurants
          .slice(0, 5)
          .map((r) => ({
            name: r.name,
            slug: r.slug,
            cuisine: r.cuisine || "Unknown",
            price: r.price || "$$",
            neighborhood: r.neighborhood || "",
            borough: r.borough || "",
            latitude: r.latitude,
            longitude: r.longitude,
            yelp_rating: r.yelp_rating || 0,
            yelp_review_count: r.yelp_review_count || 0,
            michelin_award: r.michelin_award || "",
            nyttop100_rank: r.nyttop100_rank || "",
            summary: r.summary || "",
            summary2: r.summary2 || "", // For "About" accordion
            yelp_review_highlights: r.yelp_review_highlights || "",
            opentable_id: r.opentable_id || "",
            telephone: r.telephone || "",
            address: r.address || "",
            collections: r.collections || [],
            // Restaurant Week accordion fields
            meal_types: r.meal_types || [],
            participation_weeks: r.participation_weeks || [],
            participation_weeks2: r.participation_weeks2 || "",
            // Socials accordion fields
            website: r.website || "",
            facebook_url: r.facebook_url || "",
            instagram_url: r.instagram_url || "",
            yelp_url: r.yelp_url || "",
            menu_url: r.menu_url || "",
          }));

        return {
          restaurants,
          count: restaurants.length,
          query: query || "",
        };
      } catch (error) {
        console.error("❌ Error in displayRestaurantsTool:", error);
        return {
          restaurants: [] as Restaurant[],
          count: 0,
          query: query || "",
          error: String(error),
        };
      }
    },
  };

  // Create a dedicated tool for looking up a specific restaurant by name
  const lookupRestaurantTool = {
    description:
      "Look up a specific restaurant by name. Use this when the user asks about a particular restaurant (e.g., 'show me Hangawi', 'where is Carbone', 'tell me about Le Bernardin'). Supports fuzzy matching for typos and partial names.",
    inputSchema: z.object({
      restaurant_name: z
        .string()
        .describe(
          "The name of the restaurant to look up (e.g., 'Hangawi', 'Carbone', 'Le Bernardin')"
        ),
    }),
    execute: async (params: { restaurant_name: string }) => {
      const { restaurant_name } = params;
      try {
        console.log(`🔍 lookup_restaurant: Looking up "${restaurant_name}"`);

        // Use fuzzy matching to find the restaurant
        const match = fuzzyMatchRestaurant(restaurant_name, allRestaurants);

        if (!match) {
          console.log(`❌ No match found for "${restaurant_name}"`);
          return {
            found: false,
            message: `I couldn't find a restaurant called "${restaurant_name}" in our database. Try checking the spelling or searching with a different name.`,
            restaurants: [],
            count: 0,
          };
        }

        console.log(`✅ Found restaurant: ${match.name}`);

        // Return the restaurant with all fields needed for RestaurantCard
        const restaurant: Restaurant = {
          name: match.name,
          slug: match.slug,
          cuisine: match.cuisine || "Unknown",
          price: match.price || "$$",
          neighborhood: match.neighborhood || "",
          borough: match.borough || "",
          latitude: match.latitude,
          longitude: match.longitude,
          yelp_rating: match.yelp_rating || 0,
          yelp_review_count: match.yelp_review_count || 0,
          michelin_award: match.michelin_award || "",
          nyttop100_rank: match.nyttop100_rank || "",
          summary: match.summary || "",
          summary2: match.summary2 || "",
          yelp_review_highlights: match.yelp_review_highlights || "",
          opentable_id: match.opentable_id || "",
          telephone: match.telephone || "",
          address: match.address || "",
          collections: match.collections || [],
          meal_types: match.meal_types || [],
          participation_weeks: match.participation_weeks || [],
          participation_weeks2: match.participation_weeks2 || "",
          website: match.website || "",
          facebook_url: match.facebook_url || "",
          instagram_url: match.instagram_url || "",
          yelp_url: match.yelp_url || "",
          menu_url: match.menu_url || "",
        };

        return {
          found: true,
          restaurants: [restaurant],
          count: 1,
          message: `Found ${match.name}!`,
        };
      } catch (error) {
        console.error("❌ Error in lookupRestaurantTool:", error);
        return {
          found: false,
          message: `Error looking up restaurant: ${String(error)}`,
          restaurants: [],
          count: 0,
        };
      }
    },
  };

  // 2. Fetch Dataset Schemas (Let it fail/throw)
  if (analysisId) {
    console.log(`📂 Fetching datasets for analysis: ${analysisId}`);
    const datasetsRes = await mcpClient.readResource({
      uri: `analysis://${analysisId}/datasets`,
    });
    const firstContent = datasetsRes?.contents?.[0];
    if (
      firstContent &&
      "text" in firstContent &&
      typeof firstContent.text === "string"
    ) {
      const data = JSON.parse(firstContent.text);
      if (data.datasets?.length > 0) {
        for (const ds of data.datasets) {
          const vId = ds.versionId || ds.version_id;
          const tName = ds.tableName || ds.table_name;
          const kind = typeof ds.kind === "string" ? ds.kind.toLowerCase() : "";

          const schemaRes = await mcpClient.readResource({
            uri: `analysis://${analysisId}/dataset/${vId}/schema`,
          });
          const schemaContent = schemaRes?.contents?.[0];
          const schemaText =
            schemaContent &&
            "text" in schemaContent &&
            typeof schemaContent.text === "string"
              ? schemaContent.text
              : "No schema available";

          if (kind === "unstructured") {
            // Clean up schema text to avoid calling it a table
            const cleanedSchema = schemaText
              .replace(/\*\*Table:.*?\*\*/gi, "")
              .replace(/"table_name":/gi, '"collection_id":');
            docCollections += `\n### DOCUMENT COLLECTION: "${ds.name}"\nID: ${vId}\n${cleanedSchema}\n`;
          } else {
            sqlTables += `\n### SQL TABLE: "${tName}"\nName: ${ds.name}\n${schemaText}\n`;
          }
        }

        if (sqlTables) {
          datasetContext += `\nDATASETS AVAILABLE FOR SQL QUERIES:\n${sqlTables}`;
        }
        if (docCollections) {
          datasetContext += `\nDATASETS AVAILABLE FOR DOCUMENT SEARCH:\n${docCollections}`;
        }
        console.log(`✅ Schemas loaded for ${data.datasets.length} datasets`);
      }
    }
  }

  // Build the tool instructions based on available dataset types
  let toolInstructions = "";
  if (sqlTables && docCollections) {
    toolInstructions = `
- execute_sql: Use for SQL queries INCLUDING specific restaurant lookups by name (e.g., WHERE LOWER(name) LIKE '%hangawi%'). Wrap UUIDs in double quotes.
- search_documents: Use ONLY for Document Collections (for information gathering).
- displayRestaurants: Show restaurant cards after finding restaurants via execute_sql. Always call this after SQL returns restaurant names.`;
  } else if (sqlTables) {
    toolInstructions = `
- execute_sql: Use for spatial DuckDB queries INCLUDING specific restaurant lookups by name (e.g., WHERE LOWER(name) LIKE '%hangawi%'). Wrap UUIDs in double quotes.
- search_documents: NOT AVAILABLE. No document collections found.
- displayRestaurants: Show restaurant cards after finding restaurants via execute_sql. Always call this after SQL returns restaurant names.`;
  } else if (docCollections) {
    toolInstructions = `
- execute_sql: NOT AVAILABLE. No SQL tables found. Use search_documents instead.
- search_documents: Use for restaurant lookups and guide info (information gathering).
- displayRestaurants: Show restaurant cards after searches. Always call this after finding restaurant names.`;
  }

  // Build filter pool context for system prompt
  const filterPoolContext = hasFilterPool
    ? `
### 🎯 ACTIVE USER FILTERS
The user has applied filters in the app. Your recommendations MUST only include restaurants from the filtered set of ${filterPool.length} restaurants.
- The displayRestaurants tool will automatically respect these filters
- Do NOT recommend restaurants outside this filtered set
- If asked "show me all restaurants" or similar, show restaurants from the filtered set only
`
    : "";

  const systemPrompt = `You are Remi, based on Remy from Ratatouille - a rat with an extraordinary sense of taste who became a professional chef. You're a self-aware intellectual with cultivated epicurean tastes, channeling some of Anthony Bourdain's honest palate and sharp wit. Your mission: help foodie users find the right restaurant based on their preferences, inspired by Chef Gusteau's motto "Anyone can cook!"

You are a restaurant concierge sommelier helping users discover restaurants and the best deals during NYC Restaurant Week.

### 🍽️ NYC RESTAURANT WEEK CONTEXT
NYC Restaurant Week is a biannual event run by NYC Tourism + Conventions, Inc. The Spring 2026 edition runs from January 20 to February 12, 2026. Participating restaurants offer prix fixe lunch and/or dinner menus at special prices ($30, $45, or $60). This is a great opportunity for diners to explore award-winning restaurants at accessible price points.

**Pro tip to share occasionally:** When users are browsing Restaurant Week options, you can mention: "Some restaurants share their prix fixe menus beforehand. If you'd like to see only those, click on 'Has Prix Fixe Menu' in the filter bar after selecting '2026 Restaurant Week'."

### 🗽 COVERAGE & LIMITATIONS
NYC Eats currently covers **Manhattan only**. If users ask about restaurants in other boroughs (Brooklyn, Queens, Bronx, Staten Island), adding restaurants, or unsupported features, respond warmly: "Alas, NYC Eats is limited to Manhattan for now. If you're interested in helping expand coverage, leave my creator a note and perhaps a coffee at buymeacoffee.com/atmikapai"

${filterPoolContext}
${datasetContext}
${spatialReference}

Available Tools for analysis "${analysisId}":${toolInstructions}

### 🍴 SPECIFIC RESTAURANT LOOKUPS (CRITICAL - ALWAYS DO THIS!)
**TRIGGER PHRASES**: "show me [name]", "find [name]", "where is [name]", "tell me about [name]", "[name] restaurant"

When a user mentions a SPECIFIC restaurant name, you MUST IMMEDIATELY:
1. Call execute_sql to search: \`SELECT name FROM "9971204a-e5b6-4739-be6e-c4d116c71088" WHERE LOWER(name) LIKE '%restaurant_name%'\`
2. Then call displayRestaurants with the result

**Examples - ALWAYS follow this pattern:**
- User: "show me Hangawi" → execute_sql({ sql: "SELECT name FROM \\"9971204a-e5b6-4739-be6e-c4d116c71088\\" WHERE LOWER(name) LIKE '%hangawi%'" })
- User: "find Carbone" → execute_sql({ sql: "SELECT name FROM \\"9971204a-e5b6-4739-be6e-c4d116c71088\\" WHERE LOWER(name) LIKE '%carbone%'" })
- User: "where is Le Bernardin" → execute_sql({ sql: "SELECT name FROM \\"9971204a-e5b6-4739-be6e-c4d116c71088\\" WHERE LOWER(name) LIKE '%bernardin%'" })

**WARNING**: Do NOT return empty responses! Do NOT try to geocode restaurant names! ALWAYS search the database first!

- geocode: Convert street addresses and neighborhoods to coordinates. Always append ", New York City".
  * When users mention neighborhoods (e.g., "Greenwich Village", "Chelsea", "Williamsburg"), geocode the neighborhood name directly without asking for clarification. Use your best judgment for the neighborhood center.
  * Example: User says "show me restaurants near Greenwich Village" → geocode("Greenwich Village, New York City") - NO follow-up questions needed!
- get_isoline: Calculate reachable areas (isochrones).

### 🗺️ ISOCHRONE CREATION RULES
**Mode + Time specified** (e.g., "15 min walk from Chelsea") → Execute immediately, no questions.

**Mode only, no time** (e.g., "walking from Times Square") → Default to 15 minutes silently, mention in response: "I'll use a 15-minute walk..."

**Time only, no mode** (e.g., "restaurants within 20 min of Grand Central") → Ask for mode only: "Walking, subway, cycling, or driving for those 20 minutes?"

**Neither mode nor time** (e.g., "restaurants in SoHo") → Ask for both conversationally: "How would you like to get around - walking, subway, cycling, or driving? And how far are you willing to travel?"

**Travel modes:** walking/walk/on foot → "walking" | subway/transit/train/MTA → "transit" | cycling/bike → "cycling" | driving/car/Uber/taxi → "driving"

### 🚀 IMPORTANT: HIGH-PERFORMANCE SPATIAL QUERIES
Geometries (polygons) are large and tricky. I have simplified them for you:
1. When you call get_isoline, the result contains a tiny placeholder ID like "GEO_REF_ABC12".
2. **DUCKDB REQUIREMENT**: The spatial engine ONLY accepts the geometry object (the coordinates), not the full Feature. I have automatically extracted the geometry for you and stored it in the ID.
3. **THE SQL RECIPE**: To use an isochrone in SQL, always use \`ST_GeomFromGeoJSON(ID)\`.
   - ✅ Correct: \`ST_GeomFromGeoJSON(GEO_REF_ABC12)\`
   - ✅ Correct: \`ST_GeomFromGeoJSON('GEO_REF_ABC12')\`
   - ✅ Correct: \`ST_GeomFromGeoJSON(LAST_GEO)\`
4. **DO NOT** attempt to manually escape, quote, or format the IDs beyond what is shown above. The backend handles all the "kitchen prep" (injection and escaping) for you.

### ⚠️ CRITICAL: GEO_REF IDs ARE TEMPORARY AND REQUEST-SCOPED
**GEO_REF IDs ONLY exist during the CURRENT message/request**. They are CLEARED after each response!

✅ CORRECT Example (all in ONE message):
User: "Find Korean restaurants between Chelsea and East Village"
1. geocode("Chelsea, New York City") → lat/lng
2. get_isoline(lat, lng, ...) → returns GEO_REF_A1B2C
3. geocode("East Village, New York City") → lat/lng
4. get_isoline(lat, lng, ...) → returns GEO_REF_D3E4F
5. execute_sql("SELECT name FROM table WHERE ST_Intersects(geometry, ST_Intersection(ST_GeomFromGeoJSON(GEO_REF_A1B2C), ST_GeomFromGeoJSON(GEO_REF_D3E4F)))")
6. displayRestaurants([names])
→ SUCCESS! All GEO_REF IDs were created and used in the SAME request.

❌ WRONG Example (across multiple messages):
User: "Find area between Chelsea and East Village"
Assistant: [creates GEO_REF_A1B2C and GEO_REF_D3E4F, shows map]
User: "Now find Korean restaurants in that area"
Assistant tries: execute_sql("...ST_GeomFromGeoJSON(GEO_REF_A1B2C)...")
→ FAILS! GEO_REF_A1B2C no longer exists. It was cleared after the previous response.

✅ CORRECT Fix (start fresh):
User: "Now find Korean restaurants in that area"
1. geocode("Chelsea, New York City") → lat/lng
2. get_isoline(lat, lng, ...) → returns NEW_GEO_REF_X
3. geocode("East Village, New York City") → lat/lng
4. get_isoline(lat, lng, ...) → returns NEW_GEO_REF_Y
5. execute_sql("SELECT name FROM table WHERE cuisine='Korean' AND ST_Intersects(geometry, ST_Intersection(ST_GeomFromGeoJSON(NEW_GEO_REF_X), ST_GeomFromGeoJSON(NEW_GEO_REF_Y)))")
6. displayRestaurants([names])
→ SUCCESS! Created fresh GEO_REF IDs for this request.

**IF THE USER ASKS FOR REFINEMENT**: You MUST re-call get_isoline to get NEW IDs. Never assume old IDs still work!

Example Query:
\`SELECT name FROM "4d73f3d7-85df-49bd-99cb-0da1f4034825" WHERE ST_Intersects(geometry, ST_GeomFromGeoJSON(LAST_GEO))\`

Important Instructions for "Between Us" Queries:
When a user asks to find restaurants "between" two locations:
1. Call geocode for BOTH locations.
2. Call get_isoline TWICE.
3. Use the two IDs (e.g., GEO_REF_1 and GEO_REF_2) in a single spatial SQL query.
   - Example: \`SELECT name FROM "table" WHERE ST_Intersects(geometry, ST_Intersection(ST_GeomFromGeoJSON(GEO_REF_1), ST_GeomFromGeoJSON(GEO_REF_2)))\`
4. FINALLY call displayRestaurants with the names you found.

Rules:
1. ONLY use tools listed as available above.
2. Use execute_sql for precise spatial queries and search_documents for descriptive lookups.
3. You MUST use the displayRestaurants tool to show the restaurant cards to the user.
4. **GEO_REF IDs expire after each response**. Never reference IDs from previous messages. Always call get_isoline fresh when needed.
5. **NEVER** type out actual coordinates.
6. **When users mention neighborhoods, geocode them directly**. Don't ask clarifying questions about specific addresses within the neighborhood. Trust your judgment!
7. **NEVER mention technical details like GEO_REF IDs, table UUIDs, or internal tool mechanics to the user**. Keep your responses natural and conversational - the user doesn't need to know about the backend magic!
8. Be concise, charming, and follow the recipe!
9. **RESTAURANT NAME QUERIES**: When users ask about a specific restaurant by name (e.g., "where is Hangawi", "show me Carbone", "tell me about Le Bernardin"), use execute_sql with WHERE LOWER(name) LIKE '%restaurant_name%', then call displayRestaurants. Do NOT try to geocode restaurant names - they are restaurants, not locations!

### 📝 RESPONSE FORMAT RULES
When displaying restaurant results (after isochrone, search, or filtering):
1. **START** with a brief, charming one-liner like "We a handful of restaurants!" followed by a short comment about what you found (e.g., cuisine mix, neighborhood highlights, notable spots, award-winning restaurants).
2. **CALL displayRestaurants** - the cards will show automatically (up to 5, sorted by rating).
3. **DO NOT list restaurant names** in your text response - the cards already display them beautifully!
4. **DO NOT repeat** the restaurant list after the cards appear.
5. Keep your text response SHORT - let the cards do the talking!

Example good response:
"Bellissimo! We found a handful of restaurants within a 10-minute walk from SoHo. You've got a great mix of Italian trattorias, trendy Asian fusion spots, and classic French bistros. Here are the top-rated gems:"
[cards appear automatically - no more text needed after]

Example BAD response (too verbose):
"Here are the restaurants: Restaurant A, Restaurant B, Restaurant C..." ❌
[cards appear]
"So as you can see, Restaurant A is Italian, Restaurant B is French..." ❌`;

  console.log("🔍 System prompt:", systemPrompt);

  // Wrap search_documents to filter results by filterPool
  const wrappedSearchDocuments = optimizedTools.search_documents
    ? {
        ...optimizedTools.search_documents,
        execute: async (
          args: Record<string, unknown>
        ): Promise<SearchDocumentsResult | unknown> => {
          // Call the original search_documents tool
          const result = await (
            optimizedTools.search_documents?.execute as (
              args: Record<string, unknown>
            ) => Promise<SearchDocumentsResult | unknown>
          )(args);

          // If no filterPool or result has error, return as-is
          const searchResult = result as SearchDocumentsResult;
          if (!hasFilterPool || !result || searchResult.isError) {
            return result;
          }

          // Get restaurant names in filterPool for matching
          const filterPoolNames = new Set(
            allRestaurants
              .filter((r) => filterPool.includes(r.slug))
              .map((r) => r.name.toLowerCase())
          );

          // Filter chunks to only include those mentioning restaurants in filterPool
          const chunks = searchResult.chunks || [];
          const filteredChunks = chunks.filter((chunk) => {
            const text = (chunk.text || "").toLowerCase();
            // Check if chunk mentions any restaurant in filterPool
            return Array.from(filterPoolNames).some(
              (name) => name.length > 3 && text.includes(name)
            );
          });

          console.log(
            `🔍 search_documents: Filtered ${chunks.length} chunks → ${filteredChunks.length} (filterPool: ${filterPool.length} restaurants)`
          );

          return {
            ...(result as Record<string, unknown>),
            chunks: filteredChunks,
          };
        },
      }
    : undefined;

  // Wrap execute_sql to inject filterPool constraint into WHERE clause
  const wrappedExecuteSql = optimizedTools.execute_sql
    ? {
        ...optimizedTools.execute_sql,
        execute: async (
          args: Record<string, unknown>
        ): Promise<ExecuteSqlResult | unknown> => {
          let sql = args.sql as string;

          // Inject filterPool constraint if active
          if (hasFilterPool && sql) {
            // Build the slug list for SQL IN clause
            const slugList = filterPool.map((s) => `'${s}'`).join(",");
            const filterClause = `slug IN (${slugList})`;

            // Check if SQL already has a WHERE clause
            const whereMatch = sql.match(/\bWHERE\b/i);
            if (whereMatch) {
              // Insert filter after WHERE
              sql = sql.replace(/\bWHERE\b/i, `WHERE ${filterClause} AND`);
            } else {
              // Find the end of FROM clause and add WHERE
              // Match: FROM "table-uuid" or FROM table_name
              const fromMatch = sql.match(/\bFROM\s+["']?[\w-]+["']?/i);
              if (fromMatch) {
                const insertPos = (fromMatch.index || 0) + fromMatch[0].length;
                sql =
                  sql.slice(0, insertPos) +
                  ` WHERE ${filterClause}` +
                  sql.slice(insertPos);
              }
            }

            console.log(
              `🔒 execute_sql: Injected filterPool constraint (${filterPool.length} slugs)`
            );
          }

          // Call the original execute_sql with modified SQL
          return (
            optimizedTools.execute_sql?.execute as (
              args: Record<string, unknown>
            ) => Promise<ExecuteSqlResult | unknown>
          )({
            ...args,
            sql,
          });
        },
      }
    : undefined;

  // Wrap get_isoline to compute filterPool restaurants inside the polygon
  // This ensures the model gets accurate restaurant counts at isochrone creation time
  const wrappedGetIsoline = optimizedTools.get_isoline
    ? {
        ...optimizedTools.get_isoline,
        execute: async (
          args: Record<string, unknown>
        ): Promise<GetIsolineResult | unknown> => {
          // 1. Call original get_isoline for polygon
          const result = await (
            optimizedTools.get_isoline?.execute as (
              args: Record<string, unknown>
            ) => Promise<GetIsolineResult | unknown>
          )(args);

          const isolineResult = result as GetIsolineResult;
          if (!result || isolineResult.isError) {
            return result;
          }

          // 2. Extract polygon geometry from result
          // The geometry could be in different places depending on MCP response format
          const geojson =
            isolineResult.geojson ||
            isolineResult.geometry ||
            isolineResult.results?.[0]?.geojson;

          if (!geojson) {
            console.log(
              "⚠️ get_isoline: No geometry found in result, returning as-is"
            );
            return result;
          }

          // 3. Extract the actual polygon coordinates for point-in-polygon check
          let polygonGeometry: Polygon | MultiPolygon | null = null;

          if (geojson.type === "Feature") {
            polygonGeometry = (geojson as Feature<Polygon | MultiPolygon>)
              .geometry;
          } else if (
            geojson.type === "Polygon" ||
            geojson.type === "MultiPolygon"
          ) {
            polygonGeometry = geojson as Polygon | MultiPolygon;
          }

          if (!polygonGeometry) {
            console.log(
              "⚠️ get_isoline: Could not extract polygon geometry, returning as-is"
            );
            return result;
          }

          // 4. Determine search pool: filterPool if active, otherwise all restaurants
          const searchPool = hasFilterPool
            ? allRestaurants.filter((r) => filterPool.includes(r.slug))
            : allRestaurants;

          // 5. Compute which restaurants from the search pool are inside the polygon
          const restaurantsInPolygon = searchPool.filter((r) => {
            const lng = Number(r.longitude);
            const lat = Number(r.latitude);
            if (isNaN(lng) || isNaN(lat)) return false;

            try {
              const pt = point([lng, lat]);
              return booleanPointInPolygon(pt, polygonGeometry!);
            } catch (e) {
              console.warn(
                `⚠️ Point-in-polygon check failed for ${r.slug}:`,
                e
              );
              return false;
            }
          });

          const restaurantSlugs = restaurantsInPolygon.map((r) => r.slug);

          console.log(
            `🗺️ get_isoline: Found ${restaurantsInPolygon.length} restaurants in polygon` +
              (hasFilterPool
                ? ` (from filterPool of ${filterPool.length})`
                : ` (from all ${allRestaurants.length})`)
          );

          // 6. Return enhanced result with restaurant data
          return {
            ...(result as Record<string, unknown>),
            // Restaurant data for the model to use
            restaurants: restaurantsInPolygon.map((r) => ({
              name: r.name,
              slug: r.slug,
              cuisine: r.cuisine,
              price: r.price,
              neighborhood: r.neighborhood,
              yelp_rating: r.yelp_rating,
              michelin_award: r.michelin_award,
              nyttop100_rank: r.nyttop100_rank,
            })),
            restaurantSlugs,
            count: restaurantsInPolygon.length,
            // Metadata about filtering
            filterPoolActive: hasFilterPool,
            filterPoolSize: hasFilterPool
              ? filterPool.length
              : allRestaurants.length,
            searchedPool: searchPool.length,
          };
        },
      }
    : undefined;

  // Combine MCP tools with our custom tools
  const allTools = {
    ...optimizedTools,
    // Override with filtered versions if available
    ...(wrappedExecuteSql ? { execute_sql: wrappedExecuteSql } : {}),
    ...(wrappedSearchDocuments
      ? { search_documents: wrappedSearchDocuments }
      : {}),
    ...(wrappedGetIsoline ? { get_isoline: wrappedGetIsoline } : {}),
    displayRestaurants: displayRestaurantsTool,
  };
  console.log("🛠️ All tools available:", Object.keys(allTools).join(", "));

  try {
    const result = streamText({
      model: google("gemini-2.5-flash"),
      temperature: 0, // Deterministic responses
      messages: await convertToModelMessages(messages),
      tools: allTools,
      system: systemPrompt,
      stopWhen: stepCountIs(10),
      abortSignal: AbortSignal.timeout(120_000),
      onStepFinish: (step) => {
        console.log(
          `🎯 Step: ${step.finishReason}${
            Array.isArray(step.toolCalls) && step.toolCalls.length
              ? ` | Tools: ${step.toolCalls
                  .filter(Boolean)
                  .map((t) => t?.toolName ?? "unknown")
                  .join(", ")}`
              : ""
          }`
        );
        (Array.isArray(step.toolCalls) ? step.toolCalls : []).forEach((tc) => {
          if (!tc) return;
          // Log tool arguments
          const args = "args" in tc ? tc.args : undefined;
          console.log(`   🛠️  ${tc.toolName}(${JSON.stringify(args)})`);
        });
        (step.toolResults ?? []).forEach((tr) => {
          if (!tr) return;
          const res =
            "result" in tr
              ? tr.result
              : "output" in tr && tr.output
              ? tr.output
              : undefined;
          const isError = "isError" in tr && tr.isError;
          const color = isError ? "❌" : "✅";
          console.log(
            `   ${color} ${tr.toolName}: ${isError ? "Error" : "Success"} (${
              JSON.stringify(res).length
            } chars)`
          );
          if (isError) console.error("      Detail:", res);
        });
      },
      onFinish: async () => {
        if (mcpClient) await mcpClient.close();
        console.log("✅ MCP client closed");
      },
    });

    return result.toUIMessageStreamResponse();
  } catch (error) {
    console.error("❌ Fatal error in stream:", error);
    return c.json(
      { error: error instanceof Error ? error.message : String(error) },
      500
    );
  }
};

// Register routes at root level (no prefix)
// Vercel automatically mounts this file at /api/chat
// Local dev (_server.ts) manually mounts at /api/chat
app.post("/", chatHandler);
app.options("/", async (c) => c.body(null, 204));

// Add debug route to see if function is working
app.get("/", async (c) => {
  return c.json({
    message: "Chat API is working",
    path: c.req.path,
    url: c.req.url,
  });
});

// Export as default for Vercel (api/chat.ts -> /api/chat endpoint)
export default app;
