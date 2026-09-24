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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Layer 2.5.4 §0.4.248+ — parsing tests for the seven Tlaloc sample workflow JSONs.
 *
 * <p>Each sample file under {@code maestro-server/src/test/resources/samples/sample-tlaloc-*.json}
 * is loaded from the test classpath, parsed as JSON, and structurally validated:
 *
 * <ul>
 *   <li>Top-level shape matches Maestro's canonical workflow definition (properties + workflow.id +
 *       workflow.steps).
 *   <li>Every Tlaloc-typed step has a {@code params.tlaloc} block with the canonical Tlaloc fields
 *       ({@code image}, {@code artifact_uri}, {@code manifest_ref}, {@code input_handle}, {@code
 *       output_handle_uri}).
 *   <li>The keystone typed-handoff sample carries a structurally-correct {@code
 *       SerializedBufferHandle} JSON in its consumer step's input_handle field.
 * </ul>
 *
 * <p>Real workflow execution against a live Maestro server is the L2.5.5 CI integration's concern.
 * v1 (this test) proves the JSONs are structurally valid against the canonical schema.
 */
public class TlalocSampleWorkflowsTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static final String[] SAMPLES = {
    "sample-tlaloc-program-wf",
    "sample-tlaloc-pipeline-wf",
    "sample-tlaloc-typed-handoff-wf",
    "sample-tlaloc-portfolio-wf",
    "sample-tlaloc-hpo-sweep-wf",
    "sample-tlaloc-iterative-tuning-wf",
    "sample-tlaloc-template-wf",
    "sample-tlaloc-caller-wf",
  };

  @Test
  public void allSampleWorkflowsLoadFromTestClasspath() throws Exception {
    for (String sample : SAMPLES) {
      Map<String, Object> wf = loadSample(sample);
      assertNotNull("workflow root must be non-null for " + sample, wf);
      assertNotNull("missing 'properties' in " + sample, wf.get("properties"));
      assertNotNull("missing 'workflow' in " + sample, wf.get("workflow"));
    }
  }

  @Test
  public void allSampleWorkflowsHaveOwnerAndId() throws Exception {
    for (String sample : SAMPLES) {
      Map<String, Object> wf = loadSample(sample);
      @SuppressWarnings("unchecked")
      Map<String, Object> props = (Map<String, Object>) wf.get("properties");
      @SuppressWarnings("unchecked")
      Map<String, Object> workflow = (Map<String, Object>) wf.get("workflow");
      assertEquals("tlaloc", props.get("owner"));
      assertEquals("workflow id must match filename: " + sample, sample, workflow.get("id"));
    }
  }

  @Test
  public void everyTlalocStepHasCanonicalParamsBlock() throws Exception {
    Set<String> requiredKeys =
        Set.of("image", "artifact_uri", "manifest_ref", "input_handle", "output_handle_uri");
    for (String sample : SAMPLES) {
      Map<String, Object> wf = loadSample(sample);
      List<TlalocStepRef> tlalocSteps = collectTlalocSteps(wf);
      for (TlalocStepRef ref : tlalocSteps) {
        for (String key : requiredKeys) {
          assertTrue(
              "sample " + sample + " step " + ref.id + " missing tlaloc." + key,
              ref.tlalocBlock.containsKey(key));
        }
      }
    }
  }

  @Test
  public void programSampleHasOneTlalocStep() throws Exception {
    List<TlalocStepRef> steps = collectTlalocSteps(loadSample("sample-tlaloc-program-wf"));
    assertEquals(1, steps.size());
    assertEquals("tlaloc.invoke", steps.get(0).id);
  }

  @Test
  public void pipelineSampleHasOneTlalocMiddleStage() throws Exception {
    List<TlalocStepRef> steps = collectTlalocSteps(loadSample("sample-tlaloc-pipeline-wf"));
    assertEquals(1, steps.size());
    assertEquals("tlaloc.compute", steps.get(0).id);
  }

  @Test
  public void typedHandoffSampleHasTwoTlalocSteps() throws Exception {
    List<TlalocStepRef> steps = collectTlalocSteps(loadSample("sample-tlaloc-typed-handoff-wf"));
    assertEquals(2, steps.size());
    assertEquals("tlaloc.encode", steps.get(0).id);
    assertEquals("tlaloc.score", steps.get(1).id);
  }

  @Test
  public void typedHandoffSampleConsumerCarriesValidSerializedBufferHandle() throws Exception {
    Map<String, Object> wf = loadSample("sample-tlaloc-typed-handoff-wf");
    List<TlalocStepRef> steps = collectTlalocSteps(wf);
    TlalocStepRef score = steps.get(1);
    @SuppressWarnings("unchecked")
    Map<String, Object> inputHandleParam =
        (Map<String, Object>) score.tlalocBlock.get("input_handle");
    String json = (String) inputHandleParam.get("value");
    Map<String, Object> handle =
        MAPPER.readValue(json, new TypeReference<Map<String, Object>>() {});
    // SerializedBufferHandle JSON shape: uri / contentHash / typeDescriptor / manifestRef /
    // meshName
    for (String key :
        new String[] {"uri", "contentHash", "typeDescriptor", "manifestRef", "meshName"}) {
      assertTrue(
          "typed-handoff consumer's input_handle missing '" + key + "': " + handle,
          handle.containsKey(key));
    }
    // typeDescriptor sub-object must declare dtype / dims / axisNames
    @SuppressWarnings("unchecked")
    Map<String, Object> td = (Map<String, Object>) handle.get("typeDescriptor");
    assertEquals("f32", td.get("dtype"));
    assertNotNull(td.get("dims"));
    assertNotNull(td.get("axisNames"));
  }

  @Test
  public void portfolioSampleNestsTlalocInsideForeach() throws Exception {
    List<TlalocStepRef> steps = collectTlalocSteps(loadSample("sample-tlaloc-portfolio-wf"));
    // The foreach harness contains 1 inner Tlaloc step.
    assertEquals(1, steps.size());
    assertEquals("tlaloc.shard", steps.get(0).id);
  }

  @Test
  public void hpoSweepSampleNestsTlalocInsideTwoForeachLevels() throws Exception {
    List<TlalocStepRef> steps = collectTlalocSteps(loadSample("sample-tlaloc-hpo-sweep-wf"));
    assertEquals(1, steps.size());
    assertEquals("tlaloc.train", steps.get(0).id);
  }

  @Test
  public void iterativeTuningSampleHasTlalocInWhileLoop() throws Exception {
    List<TlalocStepRef> steps = collectTlalocSteps(loadSample("sample-tlaloc-iterative-tuning-wf"));
    assertEquals(1, steps.size());
    assertEquals("tlaloc.evaluate", steps.get(0).id);
  }

  @Test
  public void templateAndCallerComposeViaSubworkflow() throws Exception {
    // Template defines a single Tlaloc step parameterised by ${input_uri} etc.
    List<TlalocStepRef> templateSteps = collectTlalocSteps(loadSample("sample-tlaloc-template-wf"));
    assertEquals(1, templateSteps.size());
    assertEquals("tlaloc.body", templateSteps.get(0).id);

    // Caller has no direct Tlaloc steps — it calls the template via subworkflow.
    Map<String, Object> caller = loadSample("sample-tlaloc-caller-wf");
    assertEquals(0, collectTlalocSteps(caller).size());
    @SuppressWarnings("unchecked")
    Map<String, Object> wf = (Map<String, Object>) caller.get("workflow");
    @SuppressWarnings("unchecked")
    List<Map<String, Object>> stepWrappers = (List<Map<String, Object>>) wf.get("steps");
    assertEquals(2, stepWrappers.size());
    for (Map<String, Object> wrapper : stepWrappers) {
      @SuppressWarnings("unchecked")
      Map<String, Object> step = (Map<String, Object>) wrapper.get("step");
      assertEquals("subworkflow", step.get("type"));
    }
  }

  // ---- Helpers ---------------------------------------------------------------

  private static Map<String, Object> loadSample(String name) throws IOException {
    String resource = "/samples/" + name + ".json";
    InputStream in = TlalocSampleWorkflowsTest.class.getResourceAsStream(resource);
    if (in == null) {
      throw new IllegalStateException("classpath resource missing: " + resource);
    }
    try (in) {
      return MAPPER.readValue(in, new TypeReference<Map<String, Object>>() {});
    }
  }

  /** Recursively descend into any nested 'steps' arrays and collect Tlaloc-typed leaf steps. */
  @SuppressWarnings("unchecked")
  private static List<TlalocStepRef> collectTlalocSteps(Map<String, Object> wfRoot) {
    List<TlalocStepRef> out = new java.util.ArrayList<>();
    Map<String, Object> workflow = (Map<String, Object>) wfRoot.get("workflow");
    if (workflow == null) return out;
    Object steps = workflow.get("steps");
    if (steps instanceof List<?>) {
      collectFromStepsList((List<Object>) steps, out);
    }
    return out;
  }

  @SuppressWarnings("unchecked")
  private static void collectFromStepsList(List<Object> list, List<TlalocStepRef> out) {
    for (Object wrapper : list) {
      if (!(wrapper instanceof Map)) continue;
      Map<String, Object> step = (Map<String, Object>) ((Map<String, Object>) wrapper).get("step");
      if (step == null) continue;
      String type = String.valueOf(step.get("type"));
      String id = String.valueOf(step.get("id"));
      if ("Tlaloc".equalsIgnoreCase(type)) {
        Map<String, Object> params = (Map<String, Object>) step.get("params");
        Map<String, Object> tlalocWrapper =
            params != null ? (Map<String, Object>) params.get("tlaloc") : null;
        Map<String, Object> tlalocBlock =
            tlalocWrapper != null ? (Map<String, Object>) tlalocWrapper.get("value") : null;
        if (tlalocBlock != null) {
          out.add(new TlalocStepRef(id, tlalocBlock));
        }
      }
      // Recurse into nested 'steps' arrays inside foreach / while / etc.
      Object nestedSteps = step.get("steps");
      if (nestedSteps instanceof List<?>) {
        collectFromStepsList((List<Object>) nestedSteps, out);
      }
    }
  }

  /** Lightweight DTO holding a Tlaloc step's id + its params.tlaloc.value block. */
  private record TlalocStepRef(String id, Map<String, Object> tlalocBlock) {}
}
