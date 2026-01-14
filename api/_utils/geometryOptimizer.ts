/**
 * Geometry optimization utilities for MCP tool responses
 * Simplifies and caches large GeoJSON geometries to reduce token usage in LLM context
 */

import * as turf from '@turf/turf';
import type { Feature, Geometry, Polygon, MultiPolygon } from 'geojson';

export type OptimizableGeometry = Feature | Polygon | MultiPolygon;

export interface OptimizedGeometry extends Feature {
  _ref_id: string;
  _optimized_geometry: string;
  _hint: string;
}

/**
 * Type guard to check if an object is an optimizable geometry
 */
export function isOptimizableGeometry(obj: unknown): obj is OptimizableGeometry {
  if (!obj || typeof obj !== 'object') return false;
  const type = (obj as Record<string, unknown>).type;
  return type === 'Feature' || type === 'Polygon' || type === 'MultiPolygon';
}

/**
 * Geometry cache for storing simplified geometries
 * Allows referencing large geometries by short IDs in SQL queries
 */
export class GeometryCache {
  private refs = new Map<string, Geometry>();
  private lastId = '';

  set(id: string, geometry: Geometry): void {
    this.refs.set(id, geometry);
    this.lastId = id;
  }

  get(id: string): Geometry | undefined {
    return this.refs.get(id);
  }

  getLastId(): string {
    return this.lastId;
  }

  has(id: string): boolean {
    return this.refs.has(id);
  }

  /**
   * Substitute geometry references in SQL with actual GeoJSON
   * Replaces GEO_REF_XXX and LAST_GEO placeholders with escaped GeoJSON strings
   */
  substituteInSQL(sql: string): string {
    let modifiedSql = sql;
    
    // Replace all geometry references
    this.refs.forEach((geo, id) => {
      const escapedGeo = JSON.stringify(geo).replace(/'/g, "''");
      const regex = new RegExp(`'${id}'|${id}`, 'g');
      modifiedSql = modifiedSql.replace(regex, `'${escapedGeo}'`);
    });

    // Replace LAST_GEO
    if (this.lastId && this.refs.has(this.lastId)) {
      const lastGeo = this.refs.get(this.lastId)!;
      const escapedLastGeo = JSON.stringify(lastGeo).replace(/'/g, "''");
      const regex = new RegExp(`'LAST_GEO'|LAST_GEO`, 'g');
      modifiedSql = modifiedSql.replace(regex, `'${escapedLastGeo}'`);
    }

    return modifiedSql;
  }
}

/**
 * Optimize a geometry by simplifying it and caching in the geometry cache
 * Returns an optimized structure with coordinates hidden and a reference ID
 * 
 * @param obj - Geometry to optimize (Feature, Polygon, or MultiPolygon)
 * @param cache - Geometry cache to store the simplified geometry
 * @returns Optimized geometry with _ref_id or original object if not optimizable
 */
export function optimizeGeometry(
  obj: unknown,
  cache: GeometryCache
): OptimizedGeometry | unknown {
  if (!isOptimizableGeometry(obj)) return obj;

  // Simplify the geometry (tolerance 0.0001 is about 10m precision)
  const simplified = turf.simplify(obj as Feature | Polygon | MultiPolygon, {
    tolerance: 0.0001,
    highQuality: false,
  });

  // Generate unique ID
  const id = `GEO_REF_${Math.random().toString(36).substring(2, 7).toUpperCase()}`;

  // CRITICAL: DuckDB ST_GeomFromGeoJSON only accepts GEOMETRY objects,
  // not full Features with properties. Cache only the geometry part.
  const geometryToCache = obj.type === 'Feature' 
    ? (simplified as Feature).geometry 
    : simplified as Polygon | MultiPolygon;

  cache.set(id, geometryToCache);

  // Return optimized structure for the UI
  // Note: The UI needs the full feature to render on the map,
  // but we hide the coordinates from the LLM context.
  return {
    ...simplified,
    _ref_id: id,
    _optimized_geometry: "Coordinates hidden to save space. Use the _ref_id variable in SQL.",
    _hint: `Geometry too large for context. Replaced with ${id}. Use ${id} in SQL queries.`,
  } as OptimizedGeometry;
}

/**
 * Recursively find and optimize all geometries in an object
 * Processes nested objects and arrays
 * 
 * @param obj - Object to process
 * @param cache - Geometry cache
 * @returns Processed object with optimized geometries
 */
export function recursivelyOptimizeGeometries(
  obj: unknown,
  cache: GeometryCache
): unknown {
  if (!obj || typeof obj !== 'object') return obj;

  // Try to optimize this object
  const optimized = optimizeGeometry(obj, cache);
  if (optimized !== obj) return optimized;

  // Recurse through arrays
  if (Array.isArray(obj)) {
    return obj.map(item => recursivelyOptimizeGeometries(item, cache));
  }

  // Recurse through object properties
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    result[key] = recursivelyOptimizeGeometries(value, cache);
  }
  return result;
}

