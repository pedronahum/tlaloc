/*
 * Copyright 2025 Tlaloc / Netflix, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package com.netflix.maestro.models.stepruntime;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.MaestroBaseTest;
import java.util.Map;
import org.junit.Test;

/**
 * Layer 3 §0.4.259+ — round-trip tests for the {@link KubernetesCommand} divergence (added {@code
 * nodeSelector} + {@code accelerators} maps).
 *
 * <p>Pin: omitted-on-null serialization (Jackson {@code @JsonInclude(NON_NULL)}), populated-fields
 * round-trip, builder defaults stay null.
 */
public class KubernetesCommandTlalocFieldsTest extends MaestroBaseTest {

  @Test
  public void unsetTlalocFieldsAreOmittedFromJson() throws Exception {
    // Pre-Tlaloc usage — neither field set. Both must be omitted from JSON
    // (otherwise a vanilla Maestro consumer that doesn't know about
    // nodeSelector/accelerators would barf on unknown nulls).
    KubernetesCommand cmd =
        KubernetesCommand.builder().appName("plain").image("plain:latest").build();
    String json = MAPPER.writeValueAsString(cmd);
    assertFalse("nodeSelector must not appear when unset; got: " + json, json.contains("node_selector"));
    assertFalse("accelerators must not appear when unset; got: " + json, json.contains("accelerators"));
  }

  @Test
  public void populatedTlalocFieldsRoundTrip() throws Exception {
    KubernetesCommand cmd =
        KubernetesCommand.builder()
            .appName("attention_step")
            .image("tlaloc-runtime:0.0.1")
            .cpu("8")
            .memory("64Gi")
            .gpu("1")
            .nodeSelector(Map.of("accelerator", "nvidia-tesla-h100"))
            .accelerators(
                Map.of(
                    "vendor", "nvidia",
                    "arch", "h100",
                    "kernel", "flash_attn_v3",
                    "kv_quant_dtype", "fp8_e4m3"))
            .build();

    String json = MAPPER.writeValueAsString(cmd);
    assertTrue("populated nodeSelector key required: " + json, json.contains("node_selector"));
    assertTrue("populated accelerators key required: " + json, json.contains("accelerators"));

    KubernetesCommand parsed = MAPPER.readValue(json, KubernetesCommand.class);
    assertEquals(cmd, parsed);
    assertEquals("nvidia-tesla-h100", parsed.getNodeSelector().get("accelerator"));
    assertEquals("flash_attn_v3", parsed.getAccelerators().get("kernel"));
    assertEquals("fp8_e4m3", parsed.getAccelerators().get("kv_quant_dtype"));
  }

  @Test
  public void existingFixturesStillRoundTripWithNewFields() throws Exception {
    // The two existing JSON fixtures (kubernetes_command.json + _exec.json)
    // don't carry node_selector / accelerators. After the divergence, they
    // should still parse + serialize back to the same shape.
    KubernetesCommand cmd =
        loadObject("fixtures/stepruntime/kubernetes_command.json", KubernetesCommand.class);
    assertNull("existing fixture must not have nodeSelector", cmd.getNodeSelector());
    assertNull("existing fixture must not have accelerators", cmd.getAccelerators());

    // Re-serialize and confirm omission.
    String json = MAPPER.writeValueAsString(cmd);
    assertFalse(json.contains("node_selector"));
    assertFalse(json.contains("accelerators"));
  }

  @Test
  public void builderToBuilderPreservesTlalocFields() {
    KubernetesCommand original =
        KubernetesCommand.builder()
            .image("img")
            .nodeSelector(Map.of("accelerator", "h100"))
            .accelerators(Map.of("vendor", "nvidia"))
            .build();
    KubernetesCommand modified = original.toBuilder().cpu("16").build();
    assertEquals(original.getNodeSelector(), modified.getNodeSelector());
    assertEquals(original.getAccelerators(), modified.getAccelerators());
    assertEquals("16", modified.getCpu());
  }
}
