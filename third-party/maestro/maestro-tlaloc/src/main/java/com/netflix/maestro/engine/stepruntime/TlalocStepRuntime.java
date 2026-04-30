/*
 * Copyright 2025 Tlaloc / Netflix, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
import com.netflix.maestro.metrics.MaestroMetrics;
import com.netflix.maestro.models.stepruntime.KubernetesCommand;
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
 * io.tlaloc.maestro.MaestroDescriptor.emit}) with proper step-type registration. The
 * descriptor emitter for the new type lands in L2.5.4+; until then this class is wired in DI
 * but only exercised by the smoke test.
 */
@Slf4j
public class TlalocStepRuntime extends KubernetesStepRuntime {

  private final TlalocEntrypointBuilder entrypointBuilder;

  /** Constructor — args mirror {@link KubernetesStepRuntime}'s + the Tlaloc entrypoint builder. */
  public TlalocStepRuntime(
      KubernetesRuntimeExecutor runtimeExecutor,
      KubernetesCommandGenerator commandGenerator,
      JobTemplateManager jobTemplateManager,
      OutputDataManager outputDataManager,
      ObjectMapper objectMapper,
      MaestroMetrics metrics,
      TlalocEntrypointBuilder entrypointBuilder) {
    super(
        runtimeExecutor,
        commandGenerator,
        jobTemplateManager,
        outputDataManager,
        objectMapper,
        metrics);
    this.entrypointBuilder = entrypointBuilder;
  }

  @Override
  protected void customizePreLaunchCommand(KubernetesStepContext context) {
    super.customizePreLaunchCommand(context);

    TlalocCommand tlalocCommand = entrypointBuilder.generateTlalocRuntime(context);

    KubernetesCommand originalCommand = context.getCommand();
    context.setCommand(
        originalCommand.toBuilder()
            .command(new String[] {"/bin/sh", "-c"})
            .args(new String[] {tlalocCommand.entrypoint()})
            .build());

    LOG.info(
        "Prepared Tlaloc step '{}' (artifact={}, manifest={})",
        tlalocCommand.stepName(),
        tlalocCommand.artifactUri(),
        tlalocCommand.manifestRef());

    // L2.5.4+ may add a TlalocArtifact analogous to ActusArtifact for
    // pendingArtifacts collection. v1 stub omits this; the runner's
    // OutputData carries the equivalent metadata.
  }
}
