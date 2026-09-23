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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.engine.kubernetes.KubernetesStepContext;
import com.netflix.maestro.exceptions.MaestroBadRequestException;
import com.netflix.maestro.models.parameter.Parameter;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Translates a Maestro {@link KubernetesStepContext}'s parameter map into the JSON payload that
 * {@link TlalocRunner} reads from inside the K8s container.
 *
 * <p>Mirrors maestro-actus's {@code ActusParamsBuilder}. v1's payload shape:
 *
 * <pre>{@code
 * {
 *   "artifact_uri":      "<URI of StableHLO body>",
 *   "manifest_ref":      "<producer's bodyHash>",
 *   "input_handle":      "<JSON-serialized SerializedBufferHandle>",
 *   "output_handle_uri": "<URI where the runner should write the SerializedBufferHandle for the output>"
 * }
 * }</pre>
 *
 * <p>v1 stub: {@link #build(KubernetesStepContext)} returns a JSON string with placeholder values
 * when params are absent. L2.5.2 fills in the real extraction logic against a fully-typed
 * Tlaloc step's params.
 */
public final class TlalocParamsBuilder {

  private static final String PARAMS_KEY = "tlaloc";

  private final ObjectMapper objectMapper;

  public TlalocParamsBuilder(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  /**
   * Build the JSON payload string that {@link TlalocRunner} consumes from {@code TLALOC_PARAMS}
   * env var (or the first command-line arg).
   */
  public String build(KubernetesStepContext context) {
    Map<String, Object> payload = new LinkedHashMap<>();
    Map<String, Parameter> stepParams =
        context.getRuntimeSummary() == null
            ? Map.<String, Parameter>of()
            : context.getRuntimeSummary().getParams();
    Map<String, Object> tlalocParams = extractTlalocBlock(stepParams);
    payload.put("artifact_uri", asString(tlalocParams.get("artifact_uri")));
    payload.put("manifest_ref", asString(tlalocParams.get("manifest_ref")));
    payload.put("input_handle", asString(tlalocParams.get("input_handle")));
    payload.put("output_handle_uri", asString(tlalocParams.get("output_handle_uri")));
    try {
      return objectMapper.writeValueAsString(payload);
    } catch (JsonProcessingException e) {
      throw new MaestroBadRequestException(e, "Failed to serialize Tlaloc params");
    }
  }

  /** Extract the {@code tlaloc} sub-block from Maestro's flat params map. */
  @SuppressWarnings("unchecked")
  private Map<String, Object> extractTlalocBlock(Map<String, Parameter> stepParams) {
    if (stepParams == null) {
      return Map.of();
    }
    Parameter raw = stepParams.get(PARAMS_KEY);
    if (raw == null) {
      return Map.of();
    }
    Object value = raw.getValue();
    if (value instanceof Map<?, ?> m) {
      return (Map<String, Object>) m;
    }
    return Map.of();
  }

  private static String asString(Object o) {
    return o == null ? "" : String.valueOf(o);
  }
}
