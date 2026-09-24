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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.models.stepruntime.KubernetesCommand;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;

/**
 * Layer 3 §0.4.259+ — pod-spec construction tests.
 *
 * <p>Pin: per-(vendor, arch) selection, lowest-cost fallback, KubernetesCommand mutation
 * (nodeSelector + accelerators + gpu), null/empty handling, and CPU-target no-GPU semantics.
 */
public class TlalocPodSpecBuilderTest {

  private ObjectMapper mapper;
  private TlalocPodSpecBuilder builder;

  @Before
  public void setUp() {
    this.mapper = new ObjectMapper();
    this.builder = new TlalocPodSpecBuilder(mapper);
  }

  /** Five-row matrix that mirrors what `populateBackendMatrix` would produce in production. */
  private String fiveRowMatrixJson() {
    return "["
        + "{\"vendor\":\"nvidia\",\"arch\":\"h100\",\"kernelName\":\"flash_attn_v3\","
        + "\"kvQuantDtype\":\"fp8_e4m3\",\"costMicroseconds\":3.5},"
        + "{\"vendor\":\"nvidia\",\"arch\":\"a100\",\"kernelName\":\"flash_attn_v2\","
        + "\"kvQuantDtype\":\"int8\",\"costMicroseconds\":7.2},"
        + "{\"vendor\":\"google\",\"arch\":\"tpu_v5e\",\"kernelName\":\"tpu_pallas_flash_attention\","
        + "\"kvQuantDtype\":\"int8\",\"costMicroseconds\":12.5},"
        + "{\"vendor\":\"aws\",\"arch\":\"trainium2\",\"kernelName\":\"nki_flash_attention\","
        + "\"kvQuantDtype\":\"fp8_e4m3\",\"costMicroseconds\":4.8},"
        + "{\"vendor\":\"tlaloc\",\"arch\":\"cpu_generic\",\"kernelName\":null,"
        + "\"kvQuantDtype\":null,\"costMicroseconds\":520.0}"
        + "]";
  }

  @Test
  public void exactVendorArchMatchPicksTheCorrectRow() {
    Optional<BackendTargetRecord> picked =
        builder.pickTarget(fiveRowMatrixJson(), "nvidia", "h100");
    assertTrue(picked.isPresent());
    assertEquals("nvidia", picked.get().vendor());
    assertEquals("h100", picked.get().arch());
    assertEquals("flash_attn_v3", picked.get().kernelName());
  }

  @Test
  public void caseInsensitiveVendorArchMatch() {
    Optional<BackendTargetRecord> picked =
        builder.pickTarget(fiveRowMatrixJson(), "NVIDIA", "H100");
    assertTrue(picked.isPresent());
    assertEquals("h100", picked.get().arch());
  }

  @Test
  public void noExactMatchFallsBackToSameVendorLowestCost() {
    // Cluster is nvidia/v100 (not in matrix). Among nvidia rows, A100 (cost 7.2) is cheaper than
    // H100 (cost 3.5)? No — H100 is cheaper. Pick H100.
    Optional<BackendTargetRecord> picked =
        builder.pickTarget(fiveRowMatrixJson(), "nvidia", "v100");
    assertTrue(picked.isPresent());
    assertEquals("nvidia", picked.get().vendor());
    assertEquals("h100", picked.get().arch());
  }

  @Test
  public void noVendorMatchFallsBackToAbsoluteLowestCost() {
    // Cluster is a hypothetical vendor with no rows. Lowest-cost overall is H100 (3.5 us).
    Optional<BackendTargetRecord> picked =
        builder.pickTarget(fiveRowMatrixJson(), "unicorn", "purple");
    assertTrue(picked.isPresent());
    assertEquals("h100", picked.get().arch());
  }

  @Test
  public void emptyMatrixReturnsEmpty() {
    assertFalse(builder.pickTarget("[]", "nvidia", "h100").isPresent());
  }

  @Test
  public void nullMatrixJsonReturnsEmpty() {
    assertFalse(builder.pickTarget(null, "nvidia", "h100").isPresent());
  }

  @Test
  public void malformedMatrixJsonReturnsEmpty() {
    assertFalse(builder.pickTarget("not a json", "nvidia", "h100").isPresent());
  }

  @Test
  public void applyBackendTargetPopulatesNodeSelectorOnH100() {
    KubernetesCommand base =
        KubernetesCommand.builder().image("tlaloc-runtime:0.0.1").cpu("8").memory("64Gi").build();
    KubernetesCommand result =
        builder.applyBackendTarget(base, fiveRowMatrixJson(), "nvidia", "h100");
    assertNotNull(result.getNodeSelector());
    assertEquals("nvidia-tesla-h100", result.getNodeSelector().get("accelerator"));
    assertEquals("nvidia", result.getAccelerators().get("vendor"));
    assertEquals("h100", result.getAccelerators().get("arch"));
    assertEquals("flash_attn_v3", result.getAccelerators().get("kernel"));
    assertEquals("fp8_e4m3", result.getAccelerators().get("kv_quant_dtype"));
    assertEquals("any GPU target asks for 1 accelerator unit", "1", result.getGpu());
    // Existing fields preserved.
    assertEquals("tlaloc-runtime:0.0.1", result.getImage());
    assertEquals("8", result.getCpu());
  }

  @Test
  public void applyBackendTargetUsesGoogleTpuLabelKey() {
    KubernetesCommand base = KubernetesCommand.builder().image("img").build();
    KubernetesCommand result =
        builder.applyBackendTarget(base, fiveRowMatrixJson(), "google", "tpu_v5e");
    assertEquals("tpu_v5e", result.getNodeSelector().get("cloud.google.com/gke-accelerator"));
  }

  @Test
  public void applyBackendTargetUsesAwsNeuronLabelKey() {
    KubernetesCommand base = KubernetesCommand.builder().image("img").build();
    KubernetesCommand result =
        builder.applyBackendTarget(base, fiveRowMatrixJson(), "aws", "trainium2");
    assertEquals("trainium2", result.getNodeSelector().get("aws.amazon.com/neuron"));
  }

  @Test
  public void cpuGenericTargetSkipsNodeSelectorAndGpu() {
    KubernetesCommand base = KubernetesCommand.builder().image("img").cpu("4").build();
    KubernetesCommand result =
        builder.applyBackendTarget(base, fiveRowMatrixJson(), "tlaloc", "cpu_generic");
    // CPU target → no GPU resource asked, no accelerator-label nodeSelector.
    assertNull(result.getGpu());
    // The accelerators map still records what was picked, for observability.
    assertEquals("tlaloc", result.getAccelerators().get("vendor"));
    assertEquals("cpu_generic", result.getAccelerators().get("arch"));
    // No "kernel" entry — null kernelName isn't put.
    assertFalse(result.getAccelerators().containsKey("kernel"));
    // Empty selector since no template fired.
    assertTrue(result.getNodeSelector().isEmpty());
  }

  @Test
  public void emptyMatrixLeavesCommandUnchanged() {
    KubernetesCommand base = KubernetesCommand.builder().image("img").cpu("4").gpu("2").build();
    KubernetesCommand result = builder.applyBackendTarget(base, "[]", "nvidia", "h100");
    assertEquals("empty matrix → identity transform", base, result);
  }

  @Test
  public void nullMatrixLeavesCommandUnchanged() {
    KubernetesCommand base = KubernetesCommand.builder().image("img").build();
    KubernetesCommand result = builder.applyBackendTarget(base, null, "nvidia", "h100");
    assertEquals(base, result);
  }

  @Test
  public void backendTargetRecordRoundTripsThroughJackson() throws Exception {
    BackendTargetRecord original =
        new BackendTargetRecord("nvidia", "h100", "flash_attn_v3", "fp8_e4m3", 3.5);
    String json = mapper.writeValueAsString(original);
    BackendTargetRecord parsed = mapper.readValue(json, BackendTargetRecord.class);
    assertEquals(original, parsed);
  }

  @Test
  public void backendTargetRecordWithNullsRoundTrips() throws Exception {
    BackendTargetRecord original =
        new BackendTargetRecord("tlaloc", "cpu_generic", null, null, null);
    String json = mapper.writeValueAsString(original);
    BackendTargetRecord parsed = mapper.readValue(json, BackendTargetRecord.class);
    assertEquals(original, parsed);
    // Null fields omitted on serialization (per @JsonInclude(NON_NULL)).
    assertFalse(json.contains("kernelName"));
    assertFalse(json.contains("kvQuantDtype"));
    assertFalse(json.contains("costMicroseconds"));
  }
}
