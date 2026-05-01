/*
 * Copyright 2025 Tlaloc / Netflix, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package com.netflix.maestro.engine.tlaloc;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.models.stepruntime.KubernetesCommand;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import lombok.extern.slf4j.Slf4j;

/**
 * Layer 3 §0.4.259+ — translate a Tlaloc {@code BackendTarget} matrix into K8s pod-spec fields.
 *
 * <p>Reads the {@code tlaloc.backend_matrix} workflow param (a JSON-encoded {@code
 * List<BackendTarget>} produced by {@code populateBackendMatrix}) and the cluster's available
 * (vendor, arch) capability, then picks the row whose target matches and translates it into:
 *
 * <ul>
 *   <li>{@link KubernetesCommand#getNodeSelector()} — K8s label selector for accelerator-aware
 *       scheduling (e.g. {@code {"accelerator":"nvidia-tesla-h100"}}).
 *   <li>{@link KubernetesCommand#getAccelerators()} — vendor + arch hints recorded for
 *       observability (e.g. {@code {"vendor":"nvidia","arch":"h100"}}).
 *   <li>{@link KubernetesCommand#getGpu()} — count, defaulted to "1" for any GPU/TPU/Trainium
 *       target (a v1 simplification; multi-accelerator pods are L4+).
 * </ul>
 *
 * <p>Selection algorithm — v1 picks the row matching {@code (cluster.vendor, cluster.arch)}; if no
 * exact match exists, falls back to the lowest-cost row in the matrix (defensive: a workflow
 * targeted at H100 deployed on an A100 cluster still launches; observability captures the
 * mismatch). When the matrix is empty or unparseable, returns an empty modification — preserves
 * the original {@link KubernetesCommand} unchanged.
 *
 * <h2>What's NOT in v1</h2>
 *
 * <ul>
 *   <li>Heterogeneous-cluster scheduling (matrix carries 7 targets, cluster has 3 of them
 *       available, scheduler picks the cheapest of the three). Stub — matches by primary
 *       vendor/arch only.
 *   <li>GPU count from cost-model memory pressure. Always emits {@code gpu="1"} for accelerator
 *       targets.
 *   <li>Spot vs on-demand selection. The cluster operator can post-process via standard K8s tooling.
 * </ul>
 */
@Slf4j
public final class TlalocPodSpecBuilder {

  /**
   * Maps lower-case vendor → K8s nodeSelector key/value patterns we emit.
   *
   * <p>Vendors absent from the map fall through to the "no nodeSelector" path — same as the
   * {@code "tlaloc"} (CPU) case, where the workload runs on any node and we don't request a GPU
   * slot. ({@code Map.of} forbids null values, so omission is the encoding for "skip".)
   */
  private static final Map<String, NodeSelectorTemplate> VENDOR_TEMPLATES =
      Map.of(
          "nvidia",
          new NodeSelectorTemplate("accelerator", arch -> "nvidia-tesla-" + arch),
          "amd",
          new NodeSelectorTemplate("accelerator", arch -> "amd-" + arch),
          "google",
          new NodeSelectorTemplate("cloud.google.com/gke-accelerator", arch -> arch),
          "aws",
          new NodeSelectorTemplate("aws.amazon.com/neuron", arch -> arch));

  private final ObjectMapper objectMapper;

  public TlalocPodSpecBuilder(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  /**
   * Apply pod-spec selections to {@code base}. Returns a new {@link KubernetesCommand} with
   * nodeSelector/accelerators/gpu populated from the picked {@link BackendTargetRecord}; returns
   * {@code base} unchanged if no row applies.
   *
   * @param base the K8s command produced upstream (has cpu/memory/image/etc. already set).
   * @param backendMatrixJson the JSON-encoded {@code List<BackendTarget>} from the manifest. May be
   *     null/empty (e.g., for steps that didn't run the populator).
   * @param clusterVendor the cluster's primary accelerator vendor (e.g. {@code "nvidia"}). Often
   *     comes from a per-deployment env var or DI binding.
   * @param clusterArch the cluster's primary accelerator arch (e.g. {@code "h100"}).
   */
  public KubernetesCommand applyBackendTarget(
      KubernetesCommand base, String backendMatrixJson, String clusterVendor, String clusterArch) {
    Optional<BackendTargetRecord> picked =
        pickTarget(backendMatrixJson, clusterVendor, clusterArch);
    if (picked.isEmpty()) return base;
    return apply(base, picked.get());
  }

  /**
   * Pick the BackendTargetRecord row that best matches the cluster. Public for testability.
   *
   * <p>Selection: exact (vendor, arch) match wins. If no exact match, fall back to the
   * lowest-cost row that has a vendor match. If no vendor match either, fall back to the
   * absolute lowest-cost row. Returns empty when the matrix is unparseable / empty.
   */
  public Optional<BackendTargetRecord> pickTarget(
      String backendMatrixJson, String clusterVendor, String clusterArch) {
    if (backendMatrixJson == null || backendMatrixJson.isBlank()) return Optional.empty();
    List<BackendTargetRecord> rows;
    try {
      rows = objectMapper.readValue(backendMatrixJson, new TypeReference<>() {});
    } catch (Exception e) {
      LOG.warn("TlalocPodSpecBuilder: malformed backend_matrix JSON; falling back to base command", e);
      return Optional.empty();
    }
    if (rows.isEmpty()) return Optional.empty();

    Optional<BackendTargetRecord> exact =
        rows.stream()
            .filter(
                r ->
                    r.vendor() != null
                        && r.vendor().equalsIgnoreCase(clusterVendor)
                        && r.arch() != null
                        && r.arch().equalsIgnoreCase(clusterArch))
            .findFirst();
    if (exact.isPresent()) return exact;

    Optional<BackendTargetRecord> sameVendor =
        rows.stream()
            .filter(r -> r.vendor() != null && r.vendor().equalsIgnoreCase(clusterVendor))
            .min(Comparator.comparing(TlalocPodSpecBuilder::costOrInfinity));
    if (sameVendor.isPresent()) return sameVendor;

    return rows.stream().min(Comparator.comparing(TlalocPodSpecBuilder::costOrInfinity));
  }

  private KubernetesCommand apply(KubernetesCommand base, BackendTargetRecord row) {
    Map<String, String> accelerators = new LinkedHashMap<>();
    accelerators.put("vendor", row.vendor());
    accelerators.put("arch", row.arch());
    if (row.kernelName() != null) accelerators.put("kernel", row.kernelName());
    if (row.kvQuantDtype() != null) accelerators.put("kv_quant_dtype", row.kvQuantDtype());

    NodeSelectorTemplate template = VENDOR_TEMPLATES.get(row.vendor() == null ? "" : row.vendor());
    Map<String, String> nodeSelector;
    String gpu = base.getGpu();
    if (template != null && row.arch() != null) {
      nodeSelector = Map.of(template.labelKey(), template.labelValueFor().apply(row.arch()));
      // v1 simplification: any non-CPU target asks for one accelerator unit.
      if (gpu == null) gpu = "1";
    } else {
      nodeSelector = Map.of();
      // CPU target — no GPU slot requested.
    }

    return base.toBuilder()
        .nodeSelector(nodeSelector)
        .accelerators(Map.copyOf(accelerators))
        .gpu(gpu)
        .build();
  }

  private static double costOrInfinity(BackendTargetRecord r) {
    return r.costMicroseconds() == null ? Double.POSITIVE_INFINITY : r.costMicroseconds();
  }

  /**
   * Vendor-specific nodeSelector emit pattern. K8s label conventions vary by cloud — NVIDIA/GKE
   * use {@code accelerator: nvidia-tesla-h100}, GCP TPUs use {@code
   * cloud.google.com/gke-accelerator: tpu_v5e}, AWS uses {@code aws.amazon.com/neuron:
   * trainium2}. Stub mapping; production deployments may override via a cluster-config
   * dependency.
   */
  private record NodeSelectorTemplate(
      String labelKey, java.util.function.Function<String, String> labelValueFor) {}
}
