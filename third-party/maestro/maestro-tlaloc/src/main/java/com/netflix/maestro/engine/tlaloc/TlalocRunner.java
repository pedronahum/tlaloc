/*
 * Copyright 2025 Tlaloc / Netflix, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
 * on-disk wire format (magic, version, payload SHA-256 against the handle's contentHash),
 * applies an <strong>identity transform</strong> (copies input bytes to {@code output_handle_uri}),
 * and emits an OutputData JSON document containing the resulting {@code SerializedBufferHandle}.
 *
 * <p>Real Tlaloc dispatch (parse the StableHLO body from {@code artifact_uri}, invoke PJRT/IREE,
 * write the computed output) requires the runtime image's full classpath including
 * {@code io.tlaloc:maestro} and IREE/PJRT bindings. That ships in L2.5.5; the identity-transform
 * v1 proves the wire-format and runner-shell wiring end-to-end without needing the heavy runtime.
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
    String artifactUri = String.valueOf(params.getOrDefault("artifact_uri", ""));
    String manifestRef = String.valueOf(params.getOrDefault("manifest_ref", ""));
    String inputHandleJson = String.valueOf(params.getOrDefault("input_handle", ""));
    String outputHandleUri = String.valueOf(params.getOrDefault("output_handle_uri", ""));

    LOG.info("Tlaloc artifact: {}", artifactUri);
    LOG.info("Tlaloc manifest: {}", manifestRef);

    Map<String, Object> outputData = new LinkedHashMap<>();
    outputData.put("manifest_ref", manifestRef);
    outputData.put("artifact_uri", artifactUri);

    if (inputHandleJson.isEmpty() || outputHandleUri.isEmpty()) {
      // Source-step / metadata-only invocation: nothing to copy. v1 records
      // the no-op result in OutputData and returns.
      outputData.put("status", "ok");
      outputData.put("note", "no input_handle or output_handle_uri provided; runner is a no-op");
      writeOutputData(outputData, outputDataPath);
      return;
    }

    Map<String, Object> inputHandle =
        OBJECT_MAPPER.readValue(inputHandleJson, new TypeReference<Map<String, Object>>() {});
    String inputUri = requireString(inputHandle, "uri");
    String expectedContentHash = requireString(inputHandle, "contentHash");
    String meshName = requireString(inputHandle, "meshName");
    Object typeDescriptor = inputHandle.get("typeDescriptor");
    if (typeDescriptor == null) {
      throw new IllegalStateException("input_handle missing 'typeDescriptor' (uri=" + inputUri + ")");
    }

    LOG.info("Reading input handle from {}", inputUri);
    byte[] inputBytes = readScheme(inputUri);
    String validatedContentHash = validateWireFormatAndHashPayload(inputBytes);
    if (!validatedContentHash.equals(expectedContentHash)) {
      throw new IllegalStateException(
          "content hash mismatch on input handle (expected="
              + expectedContentHash
              + " got="
              + validatedContentHash
              + ", uri="
              + inputUri
              + ")");
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
    outputHandle.put("uri", outputHandleUri);
    outputHandle.put("contentHash", expectedContentHash);
    outputHandle.put("typeDescriptor", typeDescriptor);
    outputHandle.put("manifestRef", manifestRef.isEmpty() ? expectedContentHash : manifestRef);
    outputHandle.put("meshName", meshName);

    outputData.put("status", "ok");
    outputData.put("output_handle", OBJECT_MAPPER.writeValueAsString(outputHandle));
    outputData.put("note", "v1 identity transform; real dispatch ships in L2.5.5");
    writeOutputData(outputData, outputDataPath);
  }

  // ---- Wire format + IO helpers ---------------------------------------------

  /** Validate magic + version, return the SHA-256 hex of the payload bytes. */
  static String validateWireFormatAndHashPayload(byte[] bytes) {
    if (bytes.length < 16) {
      throw new IllegalStateException("file too short to be wire-format buffer (" + bytes.length + " B)");
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int magic = buf.getInt();
    if (magic != WIRE_MAGIC) {
      throw new IllegalStateException(
          "bad magic: expected 0x" + Integer.toHexString(WIRE_MAGIC) + ", got 0x" + Integer.toHexString(magic));
    }
    int version = buf.getInt();
    if (version != WIRE_VERSION) {
      throw new IllegalStateException("bad version: expected " + WIRE_VERSION + ", got " + version);
    }
    int descLen = buf.getInt();
    if (descLen < 0 || descLen > buf.remaining()) {
      throw new IllegalStateException("descLen " + descLen + " out of range");
    }
    buf.position(buf.position() + descLen);
    int payloadLen = buf.getInt();
    if (payloadLen < 0 || payloadLen > buf.remaining()) {
      throw new IllegalStateException("payloadLen " + payloadLen + " out of range");
    }
    byte[] payload = new byte[payloadLen];
    buf.get(payload);
    return sha256Hex(payload);
  }

  private static byte[] readScheme(String uri) throws Exception {
    URI parsed = new URI(uri);
    return switch (parsed.getScheme()) {
      case "file" -> Files.readAllBytes(Paths.get(parsed.getPath()));
      case "s3", "gs" -> throw new UnsupportedOperationException(
          "v1 supports file:// only; '" + parsed.getScheme() + "://' stubbed for Layer 3+");
      default -> throw new IllegalStateException("unsupported URI scheme '" + parsed.getScheme() + "' (uri=" + uri + ")");
    };
  }

  private static void writeScheme(String uri, byte[] bytes) throws Exception {
    URI parsed = new URI(uri);
    switch (parsed.getScheme()) {
      case "file" -> {
        Path path = Paths.get(parsed.getPath());
        Path parent = path.getParent();
        if (parent != null && !Files.exists(parent)) {
          Files.createDirectories(parent);
        }
        Files.write(path, bytes);
      }
      case "s3", "gs" -> throw new UnsupportedOperationException(
          "v1 supports file:// only; '" + parsed.getScheme() + "://' stubbed for Layer 3+");
      default -> throw new IllegalStateException("unsupported URI scheme '" + parsed.getScheme() + "' (uri=" + uri + ")");
    }
  }

  private static void writeOutputData(Map<String, Object> outputData, Path outputDataPath) throws Exception {
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
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      byte[] digest = md.digest(bytes);
      StringBuilder sb = new StringBuilder(digest.length * 2);
      for (byte b : digest) {
        int v = b & 0xff;
        sb.append("0123456789abcdef".charAt(v >>> 4));
        sb.append("0123456789abcdef".charAt(v & 0xf));
      }
      return sb.toString();
    } catch (Exception e) {
      throw new IllegalStateException("SHA-256 unavailable", e);
    }
  }
}
