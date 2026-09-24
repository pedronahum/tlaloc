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

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

/**
 * Java mirror of {@code io.tlaloc.maestro.BackendTarget} (the Kotlin data class living in {@code
 * maestro:commonMain} from §0.4.258). Used by {@link TlalocPodSpecBuilder} to deserialize the JSON
 * encoding of {@code ProgramManifest.backendMatrix} that {@code TlalocParamsBuilder} carries
 * through the workflow params map.
 *
 * <p>Layer 3 §0.4.259+ — purely a transport struct on the runtime side. No business logic; the L3
 * pipeline (recognize → coarsen → kernel-lower → cost) populates the rows on the Kotlin side and
 * Jackson moves the data across the language boundary.
 *
 * <p>{@code @JsonInclude(NON_NULL)} matches the Kotlin emit shape — null fields are omitted on
 * serialization, ensuring round-trip fidelity with the {@code BackendTarget.fromJson} parser.
 *
 * @param vendor lower-case vendor string ({@code "nvidia"}, {@code "google"}, ...).
 * @param arch device-specific identifier ({@code "h100"}, {@code "tpu_v5e"}, ...).
 * @param kernelName the kernel identifier the L3.3 lowering pass picked, or {@code null} for the
 *     decompose path.
 * @param kvQuantDtype KV-quant dtype tag ({@code "fp8_e4m3"}, {@code "int8"}), or {@code null}.
 * @param costMicroseconds roofline-style time estimate, or {@code null} when not computed.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonNaming(PropertyNamingStrategies.LowerCamelCaseStrategy.class)
public record BackendTargetRecord(
    String vendor, String arch, String kernelName, String kvQuantDtype, Double costMicroseconds) {}
