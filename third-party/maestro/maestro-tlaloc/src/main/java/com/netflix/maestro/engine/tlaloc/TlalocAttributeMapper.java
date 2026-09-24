/*
 * Copyright 2026 Pedro N. Rodriguez
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */
package com.netflix.maestro.engine.tlaloc;

import java.util.HashMap;
import java.util.Map;

/**
 * Maps Maestro's snake_case parameter names to Tlaloc's runtime expectations.
 *
 * <p>Mirrors maestro-actus's {@code ActusAttributeMapper}. v1 carries a small set of standard
 * Tlaloc params; user-provided params pass through unchanged.
 *
 * <p>Tlaloc's runtime image reads its job parameters as JSON in the {@code TLALOC_PARAMS}
 * environment variable. The mapper surfaces here for symmetry with the maestro-actus structure even
 * though Tlaloc has fewer naming-bridge needs (Tlaloc's own surface is camelCase already at the
 * JSON level via {@code ProgramManifest.toJson}).
 */
public final class TlalocAttributeMapper {

  /**
   * Tlaloc-recognised parameter keys. Maestro workflows declare these in their {@code params: {
   * tlaloc: {...} }} block; the mapper validates membership and passes them through verbatim.
   */
  private static final Map<String, String> CANONICAL_NAMES = canonicalNames();

  private TlalocAttributeMapper() {}

  /**
   * Translate a Maestro parameter key to its Tlaloc-runtime canonical form. Returns the input
   * unchanged when no mapping is needed.
   */
  public static String canonicalize(String maestroKey) {
    return CANONICAL_NAMES.getOrDefault(maestroKey, maestroKey);
  }

  private static Map<String, String> canonicalNames() {
    Map<String, String> m = new HashMap<>();
    // v1 set: pointers + metadata that Tlaloc's runtime image reads to locate
    // and dispatch the StableHLO body.
    m.put("artifact_uri", "tlaloc_artifact_uri");
    m.put("manifest_ref", "tlaloc_manifest_ref");
    m.put("input_handle", "tlaloc_input_handle");
    m.put("output_handle_uri", "tlaloc_output_handle_uri");
    return Map.copyOf(m);
  }
}
