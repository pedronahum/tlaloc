/*
 * Copyright 2025 Tlaloc / Netflix, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package com.netflix.maestro.engine.tlaloc;

/**
 * Immutable carrier for the shell command that {@link
 * com.netflix.maestro.engine.stepruntime.TlalocStepRuntime} hands to the K8s job, plus the
 * Tlaloc-specific metadata (artifact URI, manifest reference) that the runtime image consumes.
 *
 * <p>Mirrors maestro-actus's {@code ActusCommand} record. v1 carries the artifact URI inline;
 * Layer 3+ replaces with a content-addressed registry pointer.
 *
 * @param entrypoint shell command string passed to {@code /bin/sh -c}.
 * @param artifactUri pointer at the StableHLO body (e.g. {@code data:application/...} or {@code
 *     oci://registry/image@sha256:...}).
 * @param manifestRef the producer's {@code ProgramManifest.bodyHash} for lineage / log
 *     correlation.
 * @param stepName workflow step name; included in container logs for grep-ability.
 */
public record TlalocCommand(
    String entrypoint, String artifactUri, String manifestRef, String stepName) {}
