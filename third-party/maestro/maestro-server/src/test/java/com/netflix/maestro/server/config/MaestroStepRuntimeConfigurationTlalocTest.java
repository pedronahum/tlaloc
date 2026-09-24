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
package com.netflix.maestro.server.config;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.mockito.Mockito.mock;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.netflix.maestro.engine.kubernetes.KubernetesCommandGenerator;
import com.netflix.maestro.engine.kubernetes.KubernetesRuntimeExecutor;
import com.netflix.maestro.engine.params.OutputDataManager;
import com.netflix.maestro.engine.stepruntime.TlalocStepRuntime;
import com.netflix.maestro.engine.steps.StepRuntime;
import com.netflix.maestro.engine.templates.JobTemplateManager;
import com.netflix.maestro.engine.tlaloc.TlalocEntrypointBuilder;
import com.netflix.maestro.metrics.MaestroMetrics;
import com.netflix.maestro.models.definition.StepType;
import java.util.Map;
import org.junit.Test;

/** The Spring configuration builds the Tlaloc step runtime and registers it under TLALOC. */
public class MaestroStepRuntimeConfigurationTlalocTest {

  @Test
  public void tlalocStepRuntimeIsRegisteredUnderTheTlalocStepType() {
    MaestroStepRuntimeConfiguration config = new MaestroStepRuntimeConfiguration();
    ObjectMapper objectMapper = new ObjectMapper();
    Map<StepType, StepRuntime> stepRuntimeMap = config.stepRuntimeMap();
    TlalocEntrypointBuilder entrypointBuilder =
        config.tlalocEntrypointBuilder(config.tlalocParamsBuilder(objectMapper));

    TlalocStepRuntime runtime =
        config.tlaloc(
            stepRuntimeMap,
            mock(KubernetesRuntimeExecutor.class),
            mock(KubernetesCommandGenerator.class),
            mock(JobTemplateManager.class),
            mock(OutputDataManager.class),
            objectMapper,
            mock(MaestroMetrics.class),
            entrypointBuilder,
            config.tlalocPodSpecBuilder(objectMapper));

    assertSame(runtime, stepRuntimeMap.get(StepType.TLALOC));
    assertEquals(1, stepRuntimeMap.size());
  }
}
