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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.models.stepruntime.KubernetesCommand;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;

/**
 * §0.4.461 (G3a-2) — the distributed pod-group seam. Pins the workflow-level contract
 * (docs/MULTIHOST_DESIGN.md §5): a distributed step = N member commands off ONE base, identical
 * except the env trio ({@code TLALOC_PJRT_NODE_ID}/{@code TLALOC_PJRT_NUM_NODES}/{@code
 * TLALOC_PJRT_COORDINATOR_ADDRESS} — exactly what runtime-pjrt's {@code
 * PjrtClientOptions.resolve()} reads) and the per-member dedup key. Pure-JVM certifiable; no
 * cluster, no claim of a real multi-host run (that is G4's, gated on hardware).
 */
public class TlalocPodGroupTest {

  private TlalocPodSpecBuilder builder;

  @Before
  public void setUp() {
    this.builder = new TlalocPodSpecBuilder(new ObjectMapper());
  }

  private KubernetesCommand base() {
    return KubernetesCommand.builder()
        .appName("tlaloc-trainer")
        .image("tlaloc/trainer:0.4")
        .cpu("8")
        .memory("32Gi")
        .gpu("1")
        .env(Map.of("TLALOC_MANIFEST_SHA", "abc123"))
        .jobDeduplicationKey("wf-1-step-2")
        .build();
  }

  @Test
  public void emitsOneMemberPerNodeWithRankedEnv() {
    List<KubernetesCommand> group = builder.buildPodGroup(base(), 4, "trainer-node0.tlaloc", 8476);
    assertEquals(4, group.size());
    for (int i = 0; i < 4; i++) {
      Map<String, String> env = group.get(i).getEnv();
      assertEquals(Integer.toString(i), env.get(TlalocPodSpecBuilder.ENV_NODE_ID));
      assertEquals("4", env.get(TlalocPodSpecBuilder.ENV_NUM_NODES));
      assertEquals(
          "trainer-node0.tlaloc:8476", env.get(TlalocPodSpecBuilder.ENV_COORDINATOR_ADDRESS));
    }
  }

  @Test
  public void membersAreIdenticalExceptEnvAndDedupKey() {
    KubernetesCommand base = base();
    List<KubernetesCommand> group = builder.buildPodGroup(base, 3, "coord", 9000);
    for (KubernetesCommand member : group) {
      // Mesh consistency: same image, same command surface, same resources —
      // one program manifest for the whole group.
      assertEquals(base.getImage(), member.getImage());
      assertEquals(base.getAppName(), member.getAppName());
      assertEquals(base.getCpu(), member.getCpu());
      assertEquals(base.getMemory(), member.getMemory());
      assertEquals(base.getGpu(), member.getGpu());
    }
  }

  @Test
  public void preservesBaseEnvEntries() {
    List<KubernetesCommand> group = builder.buildPodGroup(base(), 2, "coord", 9000);
    for (KubernetesCommand member : group) {
      assertEquals("abc123", member.getEnv().get("TLALOC_MANIFEST_SHA"));
      assertEquals(4, member.getEnv().size());
    }
  }

  @Test
  public void dedupKeysAreDistinctPerMember() {
    List<KubernetesCommand> group = builder.buildPodGroup(base(), 2, "coord", 9000);
    assertEquals("wf-1-step-2-node0", group.get(0).getJobDeduplicationKey());
    assertEquals("wf-1-step-2-node1", group.get(1).getJobDeduplicationKey());
  }

  @Test
  public void nullDedupKeyStaysNull() {
    KubernetesCommand base = base().toBuilder().jobDeduplicationKey(null).build();
    List<KubernetesCommand> group = builder.buildPodGroup(base, 2, "coord", 9000);
    assertNull(group.get(0).getJobDeduplicationKey());
    assertNull(group.get(1).getJobDeduplicationKey());
  }

  @Test
  public void singleNodeGroupStillCarriesTheUniformContract() {
    // Degenerate size-1 group: env trio present so PjrtClientOptions.resolve()
    // behaves identically at every group size.
    List<KubernetesCommand> group = builder.buildPodGroup(base(), 1, "coord", 9000);
    assertEquals(1, group.size());
    Map<String, String> env = group.get(0).getEnv();
    assertEquals("0", env.get(TlalocPodSpecBuilder.ENV_NODE_ID));
    assertEquals("1", env.get(TlalocPodSpecBuilder.ENV_NUM_NODES));
    assertEquals("coord:9000", env.get(TlalocPodSpecBuilder.ENV_COORDINATOR_ADDRESS));
  }

  @Test
  public void nullBaseEnvIsTreatedAsEmpty() {
    KubernetesCommand base = base().toBuilder().env(null).build();
    List<KubernetesCommand> group = builder.buildPodGroup(base, 2, "coord", 9000);
    assertEquals(3, group.get(0).getEnv().size());
  }

  @Test
  public void composesWithBackendTargetSelection() {
    // The documented composition order: accelerator selection first, group
    // expansion second — every member inherits the selected nodeSelector.
    String matrix =
        "[{\"vendor\":\"nvidia\",\"arch\":\"h100\",\"kernelName\":\"flash_attn_v3\","
            + "\"kvQuantDtype\":\"fp8_e4m3\",\"costMicroseconds\":3.5}]";
    KubernetesCommand selected = builder.applyBackendTarget(base(), matrix, "nvidia", "h100");
    List<KubernetesCommand> group = builder.buildPodGroup(selected, 2, "coord", 9000);
    for (KubernetesCommand member : group) {
      assertEquals("nvidia-tesla-h100", member.getNodeSelector().get("accelerator"));
      assertEquals("1", member.getGpu());
    }
  }

  @Test
  public void refusesReservedEnvKeysInBase() {
    KubernetesCommand base =
        base().toBuilder().env(Map.of(TlalocPodSpecBuilder.ENV_NODE_ID, "7")).build();
    IllegalArgumentException ex =
        assertThrows(
            IllegalArgumentException.class, () -> builder.buildPodGroup(base, 2, "coord", 9000));
    assertTrue(ex.getMessage().contains(TlalocPodSpecBuilder.ENV_NODE_ID));
  }

  @Test
  public void refusesMalformedGroupParameters() {
    assertThrows(
        IllegalArgumentException.class, () -> builder.buildPodGroup(base(), 0, "coord", 9000));
    assertThrows(
        IllegalArgumentException.class, () -> builder.buildPodGroup(base(), 2, "  ", 9000));
    assertThrows(
        IllegalArgumentException.class, () -> builder.buildPodGroup(base(), 2, "coord", 0));
    assertThrows(
        IllegalArgumentException.class, () -> builder.buildPodGroup(base(), 2, "coord", 70000));
  }
}
