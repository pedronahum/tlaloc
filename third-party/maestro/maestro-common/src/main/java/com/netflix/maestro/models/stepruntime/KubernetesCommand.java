/*
 * Copyright 2025 Netflix, Inc.
 *
 * Modified by Pedro N. Rodriguez for Tlaloc, 2026: adds the nodeSelector and accelerators fields.
 * See docs/vendoring.md.
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
package com.netflix.maestro.models.stepruntime;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import java.util.Map;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;

/** Kubernetes batch job command. */
@JsonDeserialize(builder = KubernetesCommand.KubernetesCommandBuilder.class)
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder(
    value = {
      "accelerators",
      "app_name",
      "args",
      "command",
      "cpu",
      "disk",
      "gpu",
      "memory",
      "image",
      "entrypoint",
      "env",
      "job_deduplication_key",
      "node_selector",
      "owner_email"
    },
    alphabetic = true)
@Builder(toBuilder = true)
@Getter
@ToString
@EqualsAndHashCode
public class KubernetesCommand {
  private final String appName;
  private final String[] command;
  private final String[] args;
  private final String cpu;
  private final String disk;
  private final String gpu;
  private final String memory;
  private final String image;

  /**
   * Shell-form entrypoint string.
   *
   * @deprecated Use {@code command} and {@code args} instead. Kept for artifact deserialization
   *     backward compatibility.
   */
  @Deprecated private final String entrypoint;

  private final Map<String, String> env;
  private final String jobDeduplicationKey;
  private final String ownerEmail;

  /**
   * Tlaloc divergence (Layer 3 §0.4.259+). K8s node selector labels for accelerator-aware
   * scheduling — populated from {@code ProgramManifest.backendMatrix} + cluster capabilities by
   * {@code TlalocPodSpecBuilder}. Null / empty for non-Tlaloc steps; downstream Maestro logic
   * passes the map through to the K8s job spec without further interpretation.
   *
   * <p>Vendoring divergence approved as L3.6's one-edit-against-upstream per audit decision D4.
   * Tracked as OQ-Layer3-4 if/when an upstream PR becomes the right path.
   */
  private final Map<String, String> nodeSelector;

  /**
   * Tlaloc divergence (Layer 3 §0.4.259+). Vendor + arch hints (e.g. {@code {"vendor":"nvidia",
   * "arch":"h100"}}) recorded alongside the K8s command for downstream debugging + observability.
   * Distinct from {@link #gpu} (a count) and {@link #nodeSelector} (label match) — this is the
   * compile-time decision the {@code BackendTarget} row carried.
   */
  private final Map<String, String> accelerators;

  /** builder class for lombok and jackson. */
  @JsonPOJOBuilder(withPrefix = "")
  @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
  public static final class KubernetesCommandBuilder {}
}
