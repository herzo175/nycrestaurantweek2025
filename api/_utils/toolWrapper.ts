/**
 * Tool wrapper utilities for MCP tools
 * Adds geometry optimization and SQL substitution to tool execution
 */

import type { ToolSet } from 'ai';
import { GeometryCache, recursivelyOptimizeGeometries } from './geometryOptimizer.js';

/**
 * Wrap MCP tools with geometry optimization middleware
 * Pre-processes SQL queries to substitute geometry references
 * Post-processes results to simplify and cache geometries
 * 
 * @param tools - Original MCP tools
 * @param cache - Geometry cache for storing simplified geometries
 * @returns Wrapped tools with optimization middleware
 */
export function wrapToolsWithGeometryOptimization(
  tools: ToolSet,
  cache: GeometryCache
): ToolSet {
  const wrapped: ToolSet = {};

  for (const [name, tool] of Object.entries(tools)) {
    wrapped[name] = {
      ...tool,
      execute: async (args: Record<string, unknown>) => {
        // Pre-process: Handle geometry references in SQL
        // Create a new args object to avoid mutating the input
        let processedArgs = args;
        if (name === 'execute_sql' && typeof args.sql === 'string') {
          processedArgs = {
            ...args,
            sql: cache.substituteInSQL(args.sql),
          };
        }

        // Execute the actual tool with the processed args
        const result = await (tool as { execute: (args: Record<string, unknown>) => Promise<unknown> }).execute(processedArgs);

        // Post-process: Optimize geometries in results
        if (result && typeof result === 'object' && !('isError' in result && (result as { isError: boolean }).isError)) {
          return recursivelyOptimizeGeometries(result, cache);
        }

        return result;
      },
    };
  }

  return wrapped;
}

