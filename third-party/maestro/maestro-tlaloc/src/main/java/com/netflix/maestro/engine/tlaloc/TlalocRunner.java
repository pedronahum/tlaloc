/*
 * Copyright 2025 Tlaloc / Netflix, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package com.netflix.maestro.engine.tlaloc;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * CLI entrypoint inside the {@code tlaloc-runtime} container image.
 *
 * <p>Invoked by {@link TlalocEntrypointBuilder}-generated shell command:
 *
 * <pre>{@code
 * java -cp '/app/*' com.netflix.maestro.engine.tlaloc.TlalocRunner '<JSON params>' '/tmp/output.json'
 * }</pre>
 *
 * <p>Mirrors maestro-actus's {@code ActusRunner.main()} signature. v1 (L2.5.1) is a stub —
 * deserializes params, echoes them to a status file. L2.5.2 wires in real Tlaloc dispatch via
 * the {@link io.tlaloc.maestro.SerializedBufferHandle} protocol (read input handle, dispatch
 * StableHLO body, write output handle).
 *
 * <h3>Args contract</h3>
 *
 * <ul>
 *   <li>{@code args[0]} — JSON params payload (see {@link TlalocParamsBuilder} for schema).
 *   <li>{@code args[1]} — output file path; the runner writes a JSON OutputData document here.
 * </ul>
 */
@Slf4j
public final class TlalocRunner {

  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  private TlalocRunner() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException(
          "TlalocRunner expects exactly 2 args: <json-params> <output-path>; got "
              + args.length);
    }
    String paramsJson = args[0];
    Path outputPath = Path.of(args[1]);

    LOG.info("TlalocRunner starting; output → {}", outputPath);

    @SuppressWarnings("unchecked")
    Map<String, Object> params = OBJECT_MAPPER.readValue(paramsJson, Map.class);
    String artifactUri = String.valueOf(params.getOrDefault("artifact_uri", ""));
    String manifestRef = String.valueOf(params.getOrDefault("manifest_ref", ""));
    String inputHandle = String.valueOf(params.getOrDefault("input_handle", ""));
    String outputHandleUri = String.valueOf(params.getOrDefault("output_handle_uri", ""));

    LOG.info("Tlaloc artifact: {}", artifactUri);
    LOG.info("Tlaloc manifest: {}", manifestRef);
    LOG.info("Input handle:    {}", inputHandle);
    LOG.info("Output handle:   {}", outputHandleUri);

    // L2.5.2 will: parse the StableHLO body from artifactUri,
    //              read the input handle via SerializedBufferHandle,
    //              dispatch via PJRT/IREE,
    //              write the output handle.
    // L2.5.1 (this stub) emits an OutputData JSON containing the echo'd params
    // so that end-to-end smoke tests can confirm the wiring is intact.

    Map<String, Object> outputData = new LinkedHashMap<>();
    outputData.put("status", "ok");
    outputData.put("layer", "2.5.1-stub");
    outputData.put("artifact_uri", artifactUri);
    outputData.put("manifest_ref", manifestRef);
    outputData.put("input_handle", inputHandle);
    outputData.put("output_handle_uri", outputHandleUri);
    outputData.put(
        "note", "L2.5.1 stub — real Tlaloc dispatch lands in L2.5.2 (TlalocRunner end-to-end).");

    Files.writeString(outputPath, OBJECT_MAPPER.writeValueAsString(outputData));
    LOG.info("TlalocRunner wrote stub output to {}", outputPath);
  }
}
