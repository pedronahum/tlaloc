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
 * mismatch). When the matrix is empty or unparseable, returns an empty modification — preserves the
 * original {@link KubernetesCommand} unchanged.
 *
 * <h2>What's NOT in v1</h2>
 *
 * <ul>
 *   <li>Heterogeneous-cluster scheduling (matrix carries 7 targets, cluster has 3 of them
 *       available, scheduler picks the cheapest of the three). Stub — matches by primary
 *       vendor/arch only.
 *   <li>GPU count from cost-model memory pressure. Always emits {@code gpu="1"} for accelerator
 *       targets.
 *   <li>Spot vs on-demand selection. The cluster operator can post-process via standard K8s
 *       tooling.
 * </ul>
 */
@Slf4j
public final class TlalocPodSpecBuilder {

  /**
   * §0.4.461 (G3a-2) — the env contract a distributed pod-group member is launched with. The names
   * are exactly what {@code PjrtClientOptions.resolve()} (runtime-pjrt) reads, so a pod whose env
   * carries them builds its PJRT client with the right rank/group/coordinator without any other
   * plumbing. See docs/MULTIHOST_DESIGN.md.
   */
  public static final String ENV_NODE_ID = "TLALOC_PJRT_NODE_ID";

  /** See {@link #ENV_NODE_ID}. */
  public static final String ENV_NUM_NODES = "TLALOC_PJRT_NUM_NODES";

  /** See {@link #ENV_NODE_ID}. */
  public static final String ENV_COORDINATOR_ADDRESS = "TLALOC_PJRT_COORDINATOR_ADDRESS";

  /**
   * Maps lower-case vendor → K8s nodeSelector key/value patterns we emit.
   *
   * <p>Vendors absent from the map fall through to the "no nodeSelector" path — same as the {@code
   * "tlaloc"} (CPU) case, where the workload runs on any node and we don't request a GPU slot.
   * ({@code Map.of} forbids null values, so omission is the encoding for "skip".)
   */
  private static final String ACCELERATOR_LABEL = "accelerator";

  private static final int MAX_PORT = 65535;

  private static final Map<String, NodeSelectorTemplate> VENDOR_TEMPLATES =
      Map.of(
          "nvidia",
          new NodeSelectorTemplate(ACCELERATOR_LABEL, arch -> "nvidia-tesla-" + arch),
          "amd",
          new NodeSelectorTemplate(ACCELERATOR_LABEL, arch -> "amd-" + arch),
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
    if (picked.isEmpty()) {
      return base;
    }
    return apply(base, picked.get());
  }

  /**
   * Pick the BackendTargetRecord row that best matches the cluster. Public for testability.
   *
   * <p>Selection: exact (vendor, arch) match wins. If no exact match, fall back to the lowest-cost
   * row that has a vendor match. If no vendor match either, fall back to the absolute lowest-cost
   * row. Returns empty when the matrix is unparseable / empty.
   */
  public Optional<BackendTargetRecord> pickTarget(
      String backendMatrixJson, String clusterVendor, String clusterArch) {
    if (backendMatrixJson == null || backendMatrixJson.isBlank()) {
      return Optional.empty();
    }
    List<BackendTargetRecord> rows;
    try {
      rows = objectMapper.readValue(backendMatrixJson, new TypeReference<>() {});
    } catch (Exception e) {
      LOG.warn(
          "TlalocPodSpecBuilder: malformed backend_matrix JSON; falling back to base command", e);
      return Optional.empty();
    }
    if (rows.isEmpty()) {
      return Optional.empty();
    }

    Optional<BackendTargetRecord> exact =
        rows.stream()
            .filter(
                r ->
                    r.vendor() != null
                        && r.vendor().equalsIgnoreCase(clusterVendor)
                        && r.arch() != null
                        && r.arch().equalsIgnoreCase(clusterArch))
            .findFirst();
    if (exact.isPresent()) {
      return exact;
    }

    Optional<BackendTargetRecord> sameVendor =
        rows.stream()
            .filter(r -> r.vendor() != null && r.vendor().equalsIgnoreCase(clusterVendor))
            .min(Comparator.comparing(TlalocPodSpecBuilder::costOrInfinity));
    if (sameVendor.isPresent()) {
      return sameVendor;
    }

    return rows.stream().min(Comparator.comparing(TlalocPodSpecBuilder::costOrInfinity));
  }

  private KubernetesCommand apply(KubernetesCommand base, BackendTargetRecord row) {
    Map<String, String> accelerators = new LinkedHashMap<>();
    accelerators.put("vendor", row.vendor());
    accelerators.put("arch", row.arch());
    if (row.kernelName() != null) {
      accelerators.put("kernel", row.kernelName());
    }
    if (row.kvQuantDtype() != null) {
      accelerators.put("kv_quant_dtype", row.kvQuantDtype());
    }

    NodeSelectorTemplate template = VENDOR_TEMPLATES.get(row.vendor() == null ? "" : row.vendor());
    Map<String, String> nodeSelector;
    String gpu = base.getGpu();
    if (template != null && row.arch() != null) {
      nodeSelector = Map.of(template.labelKey(), template.labelValueFor().apply(row.arch()));
      // v1 simplification: any non-CPU target asks for one accelerator unit.
      if (gpu == null) {
        gpu = "1";
      }
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

  /**
   * §0.4.461 (G3a-2) — expand one accelerator-selected {@link KubernetesCommand} into a DISTRIBUTED
   * POD GROUP: {@code numNodes} member commands, identical in every field except env (each member
   * gains {@link #ENV_NODE_ID}=i, {@link #ENV_NUM_NODES}=N, {@link
   * #ENV_COORDINATOR_ADDRESS}=host:port) and the job-deduplication key (suffixed {@code -nodeN} so
   * the members never collapse into one K8s job).
   *
   * <p>The workflow-level contract (docs/MULTIHOST_DESIGN.md §5): a distributed step = one program
   * manifest, N pods, mesh-consistent — every member runs the SAME image/command over the SAME
   * manifest, and only the env trio distinguishes rank. Compose with {@link #applyBackendTarget}
   * first (accelerator selection), then expand; the group is homogeneous by construction because
   * expansion copies the already-selected base.
   *
   * <p>The coordinator address names node 0's coordination service — by convention the node-0 pod's
   * stable DNS name under a headless service ({@code <group>-node0.<service>}); the builder takes
   * it as data rather than minting K8s object names (the runner owns naming).
   *
   * <p>What v1 does NOT do, by name: no PodGroup/gang-scheduling CRD emission (Kueue/Volcano are
   * cluster-operator territory; all-or-nothing scheduling is recorded as a deployment requirement,
   * not enforced here), no per-member GPU topology spreading, no multi-slice (MegaScale) env —
   * single-slice groups only until G4 measures a real one.
   *
   * @param base the fully-built single-pod command (accelerator selection already applied).
   * @param numNodes group size; must be >= 1. Size 1 returns the degenerate one-member group (env
   *     trio still emitted, so the contract is uniform and {@code PjrtClientOptions} resolves
   *     identically at every size).
   * @param coordinatorHost DNS name or IP of node 0's coordination service.
   * @param coordinatorPort port of that service, in [1, 65535].
   * @return an immutable list of {@code numNodes} member commands, index = node id.
   */
  @SuppressWarnings("PMD.AvoidInstantiatingObjectsInLoops") // one env map per pod-group member
  public List<KubernetesCommand> buildPodGroup(
      KubernetesCommand base, int numNodes, String coordinatorHost, int coordinatorPort) {
    if (numNodes < 1) {
      throw new IllegalArgumentException("buildPodGroup: numNodes must be >= 1, got " + numNodes);
    }
    if (coordinatorHost == null || coordinatorHost.isBlank()) {
      throw new IllegalArgumentException("buildPodGroup: coordinatorHost must be non-blank");
    }
    if (coordinatorPort < 1 || coordinatorPort > MAX_PORT) {
      throw new IllegalArgumentException(
          "buildPodGroup: coordinatorPort must be in [1, 65535], got " + coordinatorPort);
    }
    Map<String, String> baseEnv = base.getEnv() == null ? Map.of() : base.getEnv();
    for (String reserved : List.of(ENV_NODE_ID, ENV_NUM_NODES, ENV_COORDINATOR_ADDRESS)) {
      if (baseEnv.containsKey(reserved)) {
        throw new IllegalArgumentException(
            "buildPodGroup: base env already carries reserved key "
                + reserved
                + " — the pod-group builder owns the distributed env trio "
                + "(refusing rather than silently overwriting a rank)");
      }
    }
    String coordinatorAddress = coordinatorHost + ":" + coordinatorPort;
    List<KubernetesCommand> members = new java.util.ArrayList<>(numNodes);
    for (int nodeId = 0; nodeId < numNodes; nodeId++) {
      Map<String, String> env = new LinkedHashMap<>(baseEnv);
      env.put(ENV_NODE_ID, Integer.toString(nodeId));
      env.put(ENV_NUM_NODES, Integer.toString(numNodes));
      env.put(ENV_COORDINATOR_ADDRESS, coordinatorAddress);
      String dedupKey =
          base.getJobDeduplicationKey() == null
              ? null
              : base.getJobDeduplicationKey() + "-node" + nodeId;
      members.add(base.toBuilder().env(Map.copyOf(env)).jobDeduplicationKey(dedupKey).build());
    }
    return List.copyOf(members);
  }

  private static double costOrInfinity(BackendTargetRecord r) {
    return r.costMicroseconds() == null ? Double.POSITIVE_INFINITY : r.costMicroseconds();
  }

  /**
   * Vendor-specific nodeSelector emit pattern. K8s label conventions vary by cloud — NVIDIA/GKE use
   * {@code accelerator: nvidia-tesla-h100}, GCP TPUs use {@code cloud.google.com/gke-accelerator:
   * tpu_v5e}, AWS uses {@code aws.amazon.com/neuron: trainium2}. Stub mapping; production
   * deployments may override via a cluster-config dependency.
   */
  private record NodeSelectorTemplate(
      String labelKey, java.util.function.Function<String, String> labelValueFor) {}
}
