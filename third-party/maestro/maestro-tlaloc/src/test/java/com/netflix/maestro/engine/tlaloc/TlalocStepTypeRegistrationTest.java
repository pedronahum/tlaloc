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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

import com.netflix.maestro.models.definition.StepType;
import org.junit.Test;

/**
 * Smoke test for Layer 2.5 §0.4.245+ step-type registration.
 *
 * <p>Verifies the minimal contract before the heavier {@link
 * com.netflix.maestro.engine.stepruntime.TlalocStepRuntime} end-to-end test lands in L2.5.2:
 *
 * <ul>
 *   <li>{@link StepType#TLALOC} exists and exposes the canonical wire string {@code "Tlaloc"}.
 *   <li>{@link StepType#create} round-trips the wire string back to the enum value.
 *   <li>The Tlaloc supporting types ({@link TlalocCommand}, {@link TlalocAttributeMapper}) load and
 *       behave as documented.
 * </ul>
 */
public class TlalocStepTypeRegistrationTest {

  @Test
  public void tlalocStepTypeExistsWithCanonicalName() {
    StepType tlaloc = StepType.TLALOC;
    assertNotNull(tlaloc);
    assertEquals("Tlaloc", tlaloc.getType());
    // Tlaloc is a leaf (executable) step, like Kubernetes / Notebook / Titus.
    assertEquals(true, tlaloc.isLeaf());
  }

  @Test
  public void tlalocStepTypeRoundTripsThroughJsonCreator() {
    // Maestro's @JsonCreator on StepType uppercases the input and routes
    // through valueOf(). "Tlaloc" → TLALOC.
    StepType parsed = StepType.create("Tlaloc");
    assertEquals(StepType.TLALOC, parsed);
  }

  @Test
  public void tlalocStepTypeIsDistinctFromKubernetes() {
    // Tlaloc step type is distinct even though TlalocStepRuntime extends
    // KubernetesStepRuntime — distinct enum entries enable the StepType ->
    // StepRuntime DI mapping in MaestroStepRuntimeConfiguration to dispatch
    // to TlalocStepRuntime (not KubernetesStepRuntime) for "Tlaloc" steps.
    assertNotEquals(StepType.KUBERNETES, StepType.TLALOC);
  }

  @Test
  public void tlalocCommandExposesAllConstructorFields() {
    TlalocCommand cmd =
        new TlalocCommand(
            "java -cp '/app/*' com.netflix.maestro.engine.tlaloc.TlalocRunner '{}' '/tmp/out.json'",
            "data:application/x-tlaloc-stablehlo;sha256=abc;base64,Zm9v",
            "abc123",
            "encode_step");
    assertEquals(
        "java -cp '/app/*' com.netflix.maestro.engine.tlaloc.TlalocRunner '{}' '/tmp/out.json'",
        cmd.entrypoint());
    assertEquals("data:application/x-tlaloc-stablehlo;sha256=abc;base64,Zm9v", cmd.artifactUri());
    assertEquals("abc123", cmd.manifestRef());
    assertEquals("encode_step", cmd.stepName());
  }

  @Test
  public void attributeMapperCanonicalizesKnownKeys() {
    assertEquals("tlaloc_artifact_uri", TlalocAttributeMapper.canonicalize("artifact_uri"));
    assertEquals("tlaloc_manifest_ref", TlalocAttributeMapper.canonicalize("manifest_ref"));
    assertEquals("tlaloc_input_handle", TlalocAttributeMapper.canonicalize("input_handle"));
    assertEquals(
        "tlaloc_output_handle_uri", TlalocAttributeMapper.canonicalize("output_handle_uri"));
  }

  @Test
  public void attributeMapperPassesUnknownKeysThrough() {
    // User-supplied params not in the canonical set survive verbatim.
    assertEquals("custom_thing", TlalocAttributeMapper.canonicalize("custom_thing"));
  }
}
