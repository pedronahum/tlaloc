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
package com.netflix.maestro.engine.stepruntime;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.engine.kubernetes.KubernetesCommandGenerator;
import com.netflix.maestro.engine.kubernetes.KubernetesRuntimeExecutor;
import com.netflix.maestro.engine.kubernetes.KubernetesStepContext;
import com.netflix.maestro.engine.params.OutputDataManager;
import com.netflix.maestro.engine.templates.JobTemplateManager;
import com.netflix.maestro.engine.tlaloc.TlalocCommand;
import com.netflix.maestro.engine.tlaloc.TlalocEntrypointBuilder;
import com.netflix.maestro.engine.tlaloc.TlalocPodSpecBuilder;
import com.netflix.maestro.metrics.MaestroMetrics;
import com.netflix.maestro.models.parameter.Parameter;
import com.netflix.maestro.models.stepruntime.KubernetesCommand;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Tlaloc step runtime. First-class step type for io.tlaloc programs (Layer 2.5 §0.4.245+).
 *
 * <p>Extends {@link KubernetesStepRuntime} to launch a Tlaloc-runtime container image as a
 * Kubernetes job, mirroring the pattern from {@link NotebookStepRuntime} /
 * pedronahum/maestro-actus's {@code ActusStepRuntime}. The container's classpath holds Tlaloc's
 * runtime jars (DTensor + StableHLO loader + {@code SerializedBufferHandle} reader); the entry
 * point is {@link com.netflix.maestro.engine.tlaloc.TlalocRunner}.
 *
 * <p>Replaces Layer 2's "Kubernetes-step masquerade" approach (the deprecated {@code
 * io.tlaloc.maestro.MaestroDescriptor.emit}) with proper step-type registration. The descriptor
 * emitter for the new type lands in L2.5.4+; until then this class is wired in DI but only
 * exercised by the smoke test.
 */
@Slf4j
public class TlalocStepRuntime extends KubernetesStepRuntime {

  private final TlalocEntrypointBuilder entrypointBuilder;
  private final TlalocPodSpecBuilder podSpecBuilder;

  /**
   * Cluster's primary accelerator (vendor, arch). Configured at deployment time. v1 reads from
   * {@code TLALOC_CLUSTER_VENDOR} / {@code TLALOC_CLUSTER_ARCH} env vars when not injected
   * directly; production deployments may use a richer cluster-config service.
   */
  private final String clusterVendor;

  private final String clusterArch;

  /** Constructor — args mirror {@link KubernetesStepRuntime}'s + the Tlaloc entrypoint builder. */
  public TlalocStepRuntime(
      KubernetesRuntimeExecutor runtimeExecutor,
      KubernetesCommandGenerator commandGenerator,
      JobTemplateManager jobTemplateManager,
      OutputDataManager outputDataManager,
      ObjectMapper objectMapper,
      MaestroMetrics metrics,
      TlalocEntrypointBuilder entrypointBuilder,
      TlalocPodSpecBuilder podSpecBuilder,
      String clusterVendor,
      String clusterArch) {
    super(
        runtimeExecutor,
        commandGenerator,
        jobTemplateManager,
        outputDataManager,
        objectMapper,
        metrics);
    this.entrypointBuilder = entrypointBuilder;
    this.podSpecBuilder = podSpecBuilder;
    this.clusterVendor = clusterVendor == null ? "" : clusterVendor;
    this.clusterArch = clusterArch == null ? "" : clusterArch;
  }

  @Override
  protected void customizePreLaunchCommand(KubernetesStepContext context) {
    super.customizePreLaunchCommand(context);

    TlalocCommand tlalocCommand = entrypointBuilder.generateTlalocRuntime(context);

    KubernetesCommand originalCommand = context.getCommand();
    KubernetesCommand withEntrypoint =
        originalCommand.toBuilder()
            .command(new String[] {"/bin/sh", "-c"})
            .args(new String[] {tlalocCommand.entrypoint()})
            .build();

    // Layer 3 §0.4.259+: if the workflow params carry a Tlaloc backend
    // matrix (populated by L3.5's `populateBackendMatrix`), translate
    // the matched row into nodeSelector + accelerators + gpu before
    // handing the command to Maestro.
    String backendMatrixJson = lookupBackendMatrixJson(context);
    KubernetesCommand finalCommand =
        podSpecBuilder.applyBackendTarget(
            withEntrypoint, backendMatrixJson, clusterVendor, clusterArch);

    context.setCommand(finalCommand);

    LOG.info(
        "Prepared Tlaloc step '{}' (artifact={}, manifest={}, cluster={}/{}, accelerators={})",
        tlalocCommand.stepName(),
        tlalocCommand.artifactUri(),
        tlalocCommand.manifestRef(),
        clusterVendor,
        clusterArch,
        finalCommand.getAccelerators());

    // L2.5.4+ may add a TlalocArtifact analogous to ActusArtifact for
    // pendingArtifacts collection. v1 stub omits this; the runner's
    // OutputData carries the equivalent metadata.
  }

  /**
   * Pull the {@code tlaloc.backend_matrix} string param out of the runtime summary. Mirrors the
   * shape used by {@link TlalocEntrypointBuilder} for {@code artifact_uri} / {@code manifest_ref}.
   * Returns {@code null} when no matrix is present (older workflows / pre-L3.5 producers); the
   * pod-spec builder treats null as "no rows applicable; pass through unchanged."
   */
  @SuppressWarnings("unchecked")
  private static String lookupBackendMatrixJson(KubernetesStepContext context) {
    if (context.getRuntimeSummary() == null) {
      return null;
    }
    Map<String, Parameter> params = context.getRuntimeSummary().getParams();
    if (params == null) {
      return null;
    }
    Parameter root = params.get("tlaloc");
    if (root == null) {
      return null;
    }
    Object v = root.getValue();
    if (!(v instanceof Map<?, ?> m)) {
      return null;
    }
    Object inner = ((Map<String, Object>) m).get("backend_matrix");
    return inner == null ? null : String.valueOf(inner);
  }
}
