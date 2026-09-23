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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/**
 * Layer 2.5.2 §0.4.247+ — direct-JVM end-to-end test for {@link TlalocRunner}.
 *
 * <p>Mirrors maestro-actus's {@code ActusRunnerEndToEndTest}: invokes {@link TlalocRunner#main}
 * in the test JVM (no K8s, no real cluster), provides a synthetic {@code SerializedBufferHandle}
 * input file, asserts the runner produces a well-formed OutputData JSON pointing at a copy on
 * the output URI, and validates the wire format on both ends.
 *
 * <p>The synthetic input is built by hand here (no dep on Tlaloc's Kotlin {@code
 * io.tlaloc.maestro:SerializedBufferHandle}) — that keeps the runner-side test classpath free of
 * Tlaloc's runtime jars. Layer 2.5.5's container image build wires up the full classpath.
 */
public class TlalocRunnerEndToEndTest {

  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  public void runnerCopiesValidatedInputBytesToOutputUri() throws Exception {
    Path inputFile = tmp.newFile("input.bin").toPath();
    Path outputFile = tmp.newFolder("out").toPath().resolve("output.bin");
    Path outputDataFile = tmp.newFile("output-data.json").toPath();

    // Synthesize an F32 buffer with the wire format Tlaloc:maestro produces.
    float[] payload = {1.0f, -2.0f, 3.5f, 4.25f};
    byte[] wireBytes = encodeWireFormat(buildDescriptor(payload.length, "Sym"), payload);
    Files.write(inputFile, wireBytes);

    String contentHash = sha256Hex(payloadBytes(payload));
    String inputHandleJson = buildHandleJson(
        inputFile.toUri().toString(),
        contentHash,
        descriptorJson(payload.length),
        "producer-step-hash",
        "Mesh0");
    String paramsJson = buildParamsJson(
        inputHandleJson,
        outputFile.toUri().toString(),
        "producer-step-hash");

    TlalocRunner.main(new String[] {paramsJson, outputDataFile.toString()});

    // Output data assertions
    Map<String, Object> outputData = MAPPER.readValue(
        Files.readString(outputDataFile), new TypeReference<Map<String, Object>>() {});
    assertEquals("ok", outputData.get("status"));
    assertEquals("producer-step-hash", outputData.get("manifest_ref"));
    assertNotNull(outputData.get("output_handle"));
    @SuppressWarnings("unchecked")
    Map<String, Object> outputHandle = MAPPER.readValue(
        (String) outputData.get("output_handle"), new TypeReference<Map<String, Object>>() {});
    assertEquals(outputFile.toUri().toString(), outputHandle.get("uri"));
    assertEquals(contentHash, outputHandle.get("contentHash"));
    assertEquals("Mesh0", outputHandle.get("meshName"));

    // Output file existence + bit-for-bit equality with input (identity transform)
    assertTrue("output file must exist after runner", Files.exists(outputFile));
    assertArrayEquals(wireBytes, Files.readAllBytes(outputFile));
  }

  @Test
  public void runnerEmitsNoOpOutputWhenInputHandleAbsent() throws Exception {
    Path outputDataFile = tmp.newFile("output-data.json").toPath();
    String paramsJson = "{\"artifact_uri\":\"data:foo\",\"manifest_ref\":\"abc\","
        + "\"input_handle\":\"\",\"output_handle_uri\":\"\"}";
    TlalocRunner.main(new String[] {paramsJson, outputDataFile.toString()});
    Map<String, Object> outputData = MAPPER.readValue(
        Files.readString(outputDataFile), new TypeReference<Map<String, Object>>() {});
    assertEquals("ok", outputData.get("status"));
    assertEquals("abc", outputData.get("manifest_ref"));
    assertTrue(((String) outputData.get("note")).contains("no input_handle"));
  }

  @Test
  public void runnerFailsOnContentHashMismatch() throws Exception {
    Path inputFile = tmp.newFile("input.bin").toPath();
    Path outputFile = tmp.newFolder("out").toPath().resolve("output.bin");
    Path outputDataFile = tmp.newFile("output-data.json").toPath();

    float[] payload = {1f, 2f, 3f};
    byte[] wireBytes = encodeWireFormat(buildDescriptor(payload.length, "Sym"), payload);
    Files.write(inputFile, wireBytes);

    // Pass the wrong contentHash in the handle; runner must reject.
    String inputHandleJson = buildHandleJson(
        inputFile.toUri().toString(),
        "0".repeat(64),  // bad hash
        descriptorJson(payload.length),
        "producer", "Mesh0");
    String paramsJson = buildParamsJson(
        inputHandleJson, outputFile.toUri().toString(), "producer");

    assertThrows(
        IllegalStateException.class,
        () -> TlalocRunner.main(new String[] {paramsJson, outputDataFile.toString()}));
  }

  @Test
  public void runnerFailsOnBadMagicBytes() throws Exception {
    Path inputFile = tmp.newFile("input.bin").toPath();
    Path outputFile = tmp.newFolder("out").toPath().resolve("output.bin");
    Path outputDataFile = tmp.newFile("output-data.json").toPath();

    // Write garbage that doesn't start with the TLAL magic.
    byte[] garbage = new byte[64];
    Files.write(inputFile, garbage);

    String inputHandleJson = buildHandleJson(
        inputFile.toUri().toString(),
        sha256Hex(new byte[0]),  // doesn't matter; runner fails on magic first
        descriptorJson(0), "producer", "Mesh0");
    String paramsJson = buildParamsJson(
        inputHandleJson, outputFile.toUri().toString(), "producer");

    assertThrows(
        IllegalStateException.class,
        () -> TlalocRunner.main(new String[] {paramsJson, outputDataFile.toString()}));
  }

  @Test
  public void runnerRejectsArgCountMismatch() {
    assertThrows(
        IllegalArgumentException.class,
        () -> TlalocRunner.main(new String[] {"only-one-arg"}));
    assertThrows(
        IllegalArgumentException.class,
        () -> TlalocRunner.main(new String[] {"a", "b", "c"}));
  }

  // ---- Wire-format helpers (mirror Tlaloc's Kotlin SerializedBufferHandle) --

  private static String descriptorJson(int dim) {
    return "{\"dtype\":\"f32\",\"dims\":[" + dim + "],\"axisNames\":[]}";
  }

  private static String buildDescriptor(int dim, String axisName) {
    return descriptorJson(dim);
  }

  private static byte[] encodeWireFormat(String descriptorJson, float[] payload) {
    byte[] descBytes = descriptorJson.getBytes(StandardCharsets.UTF_8);
    byte[] payloadBytes = payloadBytes(payload);
    int total = 4 + 4 + 4 + descBytes.length + 4 + payloadBytes.length;
    ByteBuffer buf = ByteBuffer.allocate(total).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(0x4C414C54); // "TLAL"
    buf.putInt(1);
    buf.putInt(descBytes.length);
    buf.put(descBytes);
    buf.putInt(payloadBytes.length);
    buf.put(payloadBytes);
    return buf.array();
  }

  private static byte[] payloadBytes(float[] floats) {
    byte[] out = new byte[floats.length * 4];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    for (float f : floats) buf.putFloat(f);
    return out;
  }

  private static String sha256Hex(byte[] bytes) throws Exception {
    MessageDigest md = MessageDigest.getInstance("SHA-256");
    byte[] digest = md.digest(bytes);
    StringBuilder sb = new StringBuilder(digest.length * 2);
    for (byte b : digest) {
      int v = b & 0xff;
      sb.append("0123456789abcdef".charAt(v >>> 4));
      sb.append("0123456789abcdef".charAt(v & 0xf));
    }
    return sb.toString();
  }

  private static String buildHandleJson(
      String uri, String hash, String descriptorJson, String manifestRef, String meshName)
      throws IOException {
    Map<String, Object> handle = new LinkedHashMap<>();
    handle.put("uri", uri);
    handle.put("contentHash", hash);
    handle.put("typeDescriptor", MAPPER.readValue(descriptorJson, new TypeReference<Map<String, Object>>() {}));
    handle.put("manifestRef", manifestRef);
    handle.put("meshName", meshName);
    return MAPPER.writeValueAsString(handle);
  }

  private static String buildParamsJson(String inputHandleJson, String outputHandleUri, String manifestRef)
      throws IOException {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("artifact_uri", "data:application/x-tlaloc-stablehlo;sha256=test;base64,");
    params.put("manifest_ref", manifestRef);
    params.put("input_handle", inputHandleJson);
    params.put("output_handle_uri", outputHandleUri);
    return MAPPER.writeValueAsString(params);
  }
}
