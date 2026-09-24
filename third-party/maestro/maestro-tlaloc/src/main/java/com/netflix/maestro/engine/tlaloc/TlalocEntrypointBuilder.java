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

import com.netflix.maestro.engine.kubernetes.KubernetesStepContext;
import com.netflix.maestro.models.parameter.Parameter;
import java.util.Map;

/**
 * Builds the shell entrypoint that the K8s job runs to invoke {@link TlalocRunner} inside the
 * tlaloc-runtime container image.
 *
 * <p>Mirrors maestro-actus's {@code ActusEntrypointBuilder}. The generated command looks like:
 *
 * <pre>{@code
 * java -cp '/app/*' com.netflix.maestro.engine.tlaloc.TlalocRunner '<JSON params>' '/tmp/maestro-tlaloc-output.json'
 * }</pre>
 *
 * <p>JSON params are escaped for single-quote shell embedding. The runner writes its output JSON to
 * the second arg path; Maestro collects it as {@link com.netflix.maestro.engine.dto.OutputData}.
 */
public final class TlalocEntrypointBuilder {

  /** Default output path inside the container; matches maestro-actus's convention. */
  public static final String DEFAULT_OUTPUT_PATH = "/tmp/maestro-tlaloc-output.json";

  private static final String SINGLE_QUOTE = "'";

  private final TlalocParamsBuilder paramsBuilder;

  public TlalocEntrypointBuilder(TlalocParamsBuilder paramsBuilder) {
    this.paramsBuilder = paramsBuilder;
  }

  /** Generate the {@link TlalocCommand} for [context]. */
  public TlalocCommand generateTlalocRuntime(KubernetesStepContext context) {
    String paramsJson = paramsBuilder.build(context);
    String escaped = escapeSingleQuotes(paramsJson);
    String entrypoint =
        "java -cp '/app/*' com.netflix.maestro.engine.tlaloc.TlalocRunner '"
            + escaped
            + "' '"
            + DEFAULT_OUTPUT_PATH
            + SINGLE_QUOTE;

    String artifactUri = lookupParam(context, "artifact_uri");
    String manifestRef = lookupParam(context, "manifest_ref");
    String stepName =
        context.getRuntimeSummary() != null ? context.getRuntimeSummary().getStepId() : "unknown";

    return new TlalocCommand(entrypoint, artifactUri, manifestRef, stepName);
  }

  private static String escapeSingleQuotes(String s) {
    // Same approach as maestro-actus: 'foo'\''bar' wraps a single quote.
    return s.replace(SINGLE_QUOTE, "'\\''");
  }

  @SuppressWarnings("unchecked")
  private static String lookupParam(KubernetesStepContext context, String key) {
    if (context.getRuntimeSummary() == null) {
      return "";
    }
    Map<String, Parameter> params = context.getRuntimeSummary().getParams();
    if (params == null) {
      return "";
    }
    Parameter root = params.get("tlaloc");
    if (root == null) {
      return "";
    }
    Object v = root.getValue();
    if (v instanceof Map<?, ?> m) {
      Object inner = ((Map<String, Object>) m).get(key);
      return inner == null ? "" : String.valueOf(inner);
    }
    return "";
  }
}
