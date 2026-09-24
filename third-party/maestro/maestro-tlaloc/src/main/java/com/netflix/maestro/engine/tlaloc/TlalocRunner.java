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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.HexFormat;
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
 * <p>Mirrors maestro-actus's {@code ActusRunner.main()} signature (Layer 2.5 §0.4.246+).
 *
 * <h3>v1 behaviour (Layer 2.5.2 / §0.4.247+)</h3>
 *
 * <p>Reads a {@code SerializedBufferHandle} JSON pointer at the producer's bytes, validates the
 * on-disk wire format (magic, version, payload SHA-256 against the handle's contentHash), applies
 * an <strong>identity transform</strong> (copies input bytes to {@code output_handle_uri}), and
 * emits an OutputData JSON document containing the resulting {@code SerializedBufferHandle}.
 *
 * <p>Real Tlaloc dispatch (parse the StableHLO body from {@code artifact_uri}, invoke PJRT/IREE,
 * write the computed output) requires the runtime image's full classpath including {@code
 * io.tlaloc:maestro} and IREE/PJRT bindings. That ships in L2.5.5; the identity-transform v1 proves
 * the wire-format and runner-shell wiring end-to-end without needing the heavy runtime.
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

  /** Magic bytes for the binary wire format ("TLAL" little-endian). Mirrors Tlaloc:maestro. */
  static final int WIRE_MAGIC = 0x4C414C54;

  /** Wire format version. v1 supports F32 payloads only. */
  static final int WIRE_VERSION = 1;

  /** Magic, version, descriptor length and payload length: four little-endian ints. */
  private static final int WIRE_HEADER_BYTES = 4 * Integer.BYTES;

  private static final String PARAM_ARTIFACT_URI = "artifact_uri";
  private static final String PARAM_MANIFEST_REF = "manifest_ref";
  private static final String OUT_STATUS = "status";
  private static final String OUT_NOTE = "note";
  private static final String STATUS_OK = "ok";
  private static final String HANDLE_URI = "uri";
  private static final String HANDLE_CONTENT_HASH = "contentHash";
  private static final String HANDLE_MESH_NAME = "meshName";
  private static final String HANDLE_TYPE_DESCRIPTOR = "typeDescriptor";
  private static final String SCHEME_FILE = "file";

  private TlalocRunner() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException(
          "TlalocRunner expects exactly 2 args: <json-params> <output-path>; got " + args.length);
    }
    String paramsJson = args[0];
    Path outputDataPath = Paths.get(args[1]);

    LOG.info("TlalocRunner starting; OutputData → {}", outputDataPath);

    Map<String, Object> params =
        OBJECT_MAPPER.readValue(paramsJson, new TypeReference<Map<String, Object>>() {});
    String artifactUri = String.valueOf(params.getOrDefault(PARAM_ARTIFACT_URI, ""));
    String manifestRef = String.valueOf(params.getOrDefault(PARAM_MANIFEST_REF, ""));
    String inputHandleJson = String.valueOf(params.getOrDefault("input_handle", ""));
    String outputHandleUri = String.valueOf(params.getOrDefault("output_handle_uri", ""));

    LOG.info("Tlaloc artifact: {}", artifactUri);
    LOG.info("Tlaloc manifest: {}", manifestRef);

    Map<String, Object> outputData = new LinkedHashMap<>();
    outputData.put(PARAM_MANIFEST_REF, manifestRef);
    outputData.put(PARAM_ARTIFACT_URI, artifactUri);

    if (inputHandleJson.isEmpty() || outputHandleUri.isEmpty()) {
      // Source-step / metadata-only invocation: nothing to copy. v1 records
      // the no-op result in OutputData and returns.
      outputData.put(OUT_STATUS, STATUS_OK);
      outputData.put(OUT_NOTE, "no input_handle or output_handle_uri provided; runner is a no-op");
      writeOutputData(outputData, outputDataPath);
      return;
    }

    Map<String, Object> inputHandle =
        OBJECT_MAPPER.readValue(inputHandleJson, new TypeReference<Map<String, Object>>() {});
    String inputUri = requireString(inputHandle, HANDLE_URI);
    String expectedContentHash = requireString(inputHandle, HANDLE_CONTENT_HASH);
    String meshName = requireString(inputHandle, HANDLE_MESH_NAME);
    Object typeDescriptor = inputHandle.get(HANDLE_TYPE_DESCRIPTOR);
    if (typeDescriptor == null) {
      throw new IllegalStateException(
          String.format("input_handle missing '%s' (uri=%s)", HANDLE_TYPE_DESCRIPTOR, inputUri));
    }

    LOG.info("Reading input handle from {}", inputUri);
    byte[] inputBytes = readScheme(inputUri);
    String validatedContentHash = validateWireFormatAndHashPayload(inputBytes);
    if (!validatedContentHash.equals(expectedContentHash)) {
      throw new IllegalStateException(
          String.format(
              "content hash mismatch on input handle (expected=%s got=%s, uri=%s)",
              expectedContentHash, validatedContentHash, inputUri));
    }

    // v1 identity transform: write the same bytes to the output handle URI.
    // Real dispatch (StableHLO body → PJRT/IREE → result tensor → re-serialize)
    // ships in L2.5.5 with the runtime image + io.tlaloc:maestro on classpath.
    LOG.info("Writing output handle to {} (identity transform)", outputHandleUri);
    writeScheme(outputHandleUri, inputBytes);

    // Compose the output SerializedBufferHandle JSON exactly as Tlaloc's Kotlin
    // SerializedBufferHandle.toJson() would — same field set, identical content
    // hash (payload is unchanged).
    Map<String, Object> outputHandle = new LinkedHashMap<>();
    outputHandle.put(HANDLE_URI, outputHandleUri);
    outputHandle.put(HANDLE_CONTENT_HASH, expectedContentHash);
    outputHandle.put(HANDLE_TYPE_DESCRIPTOR, typeDescriptor);
    outputHandle.put("manifestRef", manifestRef.isEmpty() ? expectedContentHash : manifestRef);
    outputHandle.put(HANDLE_MESH_NAME, meshName);

    outputData.put(OUT_STATUS, STATUS_OK);
    outputData.put("output_handle", OBJECT_MAPPER.writeValueAsString(outputHandle));
    outputData.put(OUT_NOTE, "v1 identity transform; real dispatch ships in L2.5.5");
    writeOutputData(outputData, outputDataPath);
  }

  // ---- Wire format + IO helpers ---------------------------------------------

  /** Validate magic + version, return the SHA-256 hex of the payload bytes. */
  static String validateWireFormatAndHashPayload(byte[] bytes) {
    if (bytes.length < WIRE_HEADER_BYTES) {
      throw new IllegalStateException(
          String.format("file too short to be wire-format buffer (%d B)", bytes.length));
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int magic = buf.getInt();
    if (magic != WIRE_MAGIC) {
      throw new IllegalStateException(
          "bad magic: expected 0x"
              + Integer.toHexString(WIRE_MAGIC)
              + ", got 0x"
              + Integer.toHexString(magic));
    }
    int version = buf.getInt();
    if (version != WIRE_VERSION) {
      throw new IllegalStateException("bad version: expected " + WIRE_VERSION + ", got " + version);
    }
    int descLen = checkedLength(buf, "descLen");
    buf.position(buf.position() + descLen);
    int payloadLen = checkedLength(buf, "payloadLen");
    byte[] payload = new byte[payloadLen];
    buf.get(payload);
    return sha256Hex(payload);
  }

  private static int checkedLength(ByteBuffer buf, String name) {
    int len = buf.getInt();
    if (len < 0 || len > buf.remaining()) {
      throw new IllegalStateException(name + " " + len + " out of range");
    }
    return len;
  }

  private static byte[] readScheme(String uri) throws Exception {
    URI parsed = new URI(uri);
    if (!SCHEME_FILE.equals(parsed.getScheme())) {
      throw unsupportedScheme(parsed, uri);
    }
    return Files.readAllBytes(Paths.get(parsed.getPath()));
  }

  private static void writeScheme(String uri, byte[] bytes) throws Exception {
    URI parsed = new URI(uri);
    if (!SCHEME_FILE.equals(parsed.getScheme())) {
      throw unsupportedScheme(parsed, uri);
    }
    Path path = Paths.get(parsed.getPath());
    Path parent = path.getParent();
    if (parent != null && !Files.exists(parent)) {
      Files.createDirectories(parent);
    }
    Files.write(path, bytes);
  }

  private static RuntimeException unsupportedScheme(URI parsed, String uri) {
    String scheme = parsed.getScheme();
    if ("s3".equals(scheme) || "gs".equals(scheme)) {
      return new UnsupportedOperationException(
          "v1 supports file:// only; '" + scheme + "://' stubbed for Layer 3+");
    }
    return new IllegalStateException(
        String.format("unsupported URI scheme '%s' (uri=%s)", scheme, uri));
  }

  private static void writeOutputData(Map<String, Object> outputData, Path outputDataPath)
      throws Exception {
    Files.writeString(outputDataPath, OBJECT_MAPPER.writeValueAsString(outputData));
    LOG.info("TlalocRunner wrote OutputData to {}", outputDataPath);
  }

  private static String requireString(Map<String, Object> m, String key) {
    Object v = m.get(key);
    if (v == null) {
      throw new IllegalStateException("missing required field '" + key + "' in handle: " + m);
    }
    return String.valueOf(v);
  }

  private static String sha256Hex(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (Exception e) {
      throw new IllegalStateException("SHA-256 unavailable", e);
    }
  }
}
