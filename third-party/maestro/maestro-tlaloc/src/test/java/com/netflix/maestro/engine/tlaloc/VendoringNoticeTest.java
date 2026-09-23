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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * Apache-2.0 obligations for the vendored Maestro tree: the Tlaloc repository's NOTICE credits
 * Netflix Maestro at the pinned commit, every upstream file Tlaloc changed carries a "Modified by"
 * line and is listed in docs/vendoring.md, and Tlaloc-written Java files carry Tlaloc's copyright
 * rather than Netflix's.
 */
public class VendoringNoticeTest {
  private static final String MODIFIED_MARK = "Modified by Pedro N. Rodriguez for Tlaloc";
  private static final String TLALOC_COPYRIGHT = "Copyright 2026 Pedro N. Rodriguez";

  private static Path repoRoot() {
    Path p = Paths.get("").toAbsolutePath();
    while (p != null) {
      if (Files.isRegularFile(p.resolve("settings.gradle.kts"))
          && Files.isDirectory(p.resolve("third-party/maestro"))) {
        return p;
      }
      p = p.getParent();
    }
    fail("VendoringNoticeTest must run inside the Tlaloc repository (no settings.gradle.kts above "
        + Paths.get("").toAbsolutePath() + ")");
    return null;
  }

  private static String read(Path p) throws IOException {
    return new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
  }

  private static String head(Path p, int lines) throws IOException {
    try (Stream<String> s = Files.lines(p, StandardCharsets.UTF_8)) {
      return s.limit(lines).collect(Collectors.joining("\n"));
    }
  }

  private static boolean isBuildOutput(Path rel) {
    for (Path part : rel) {
      String n = part.toString();
      if (n.equals("build") || n.equals(".gradle") || n.equals("out")) {
        return true;
      }
    }
    return false;
  }

  /** Paths in the "Changes to the upstream tree" table of docs/vendoring.md. */
  private static List<String> documentedChanges(Path root) throws IOException {
    String doc = read(root.resolve("docs/vendoring.md"));
    int start = doc.indexOf("## Changes to the upstream tree");
    assertTrue("docs/vendoring.md has no '## Changes to the upstream tree' section", start >= 0);
    int end = doc.indexOf("\n## ", start + 1);
    String section = doc.substring(start, end < 0 ? doc.length() : end);
    List<String> paths = new ArrayList<>();
    Matcher m = Pattern.compile("(?m)^\\| `([^`]+)` \\|").matcher(section);
    while (m.find()) {
      paths.add(m.group(1));
    }
    assertFalse("the changes table in docs/vendoring.md lists no files", paths.isEmpty());
    return paths;
  }

  @Test
  public void everyDocumentedChangeIsMarkedAndEveryMarkedFileIsDocumented() throws IOException {
    Path root = repoRoot();
    Path maestro = root.resolve("third-party/maestro");
    TreeSet<String> documented = new TreeSet<>(documentedChanges(root));
    for (String rel : documented) {
      Path f = maestro.resolve(rel);
      assertTrue("docs/vendoring.md lists " + rel + ", which does not exist", Files.isRegularFile(f));
      assertTrue(
          rel + " is listed as modified but has no '" + MODIFIED_MARK + "' line in its first 12 lines",
          head(f, 12).contains(MODIFIED_MARK));
    }
    TreeSet<String> marked = new TreeSet<>();
    try (Stream<Path> files = Files.walk(maestro)) {
      for (Path f :
          files
              .filter(Files::isRegularFile)
              .filter(f -> !isBuildOutput(maestro.relativize(f)))
              .filter(f -> !maestro.relativize(f).startsWith("maestro-tlaloc"))
              .filter(f -> f.toString().matches(".*\\.(java|gradle)$"))
              .collect(Collectors.toList())) {
        if (head(f, 12).contains(MODIFIED_MARK)) {
          marked.add(maestro.relativize(f).toString().replace('\\', '/'));
        }
      }
    }
    assertEquals("files carrying a 'Modified by' line vs the docs/vendoring.md table", documented, marked);
  }

  @Test
  public void noticeCreditsNetflixMaestroAtThePinnedCommit() throws IOException {
    Path root = repoRoot();
    Path notice = root.resolve("NOTICE");
    assertTrue("the repository has no root NOTICE file", Files.isRegularFile(notice));
    String text = read(notice);
    Matcher pin =
        Pattern.compile("\\| Netflix/maestro \\| `([0-9a-f]{40})` \\|")
            .matcher(read(root.resolve("docs/vendoring.md")));
    assertTrue("docs/vendoring.md names no Netflix/maestro pin", pin.find());
    assertTrue("NOTICE does not credit Netflix, Inc.", text.contains("Netflix, Inc."));
    assertTrue("NOTICE does not name Netflix Maestro", text.contains("Netflix Maestro"));
    assertTrue(
        "NOTICE does not name the pinned upstream commit " + pin.group(1),
        text.contains(pin.group(1)));
  }

  @Test
  public void tlalocWrittenJavaFilesCarryTlalocCopyright() throws IOException {
    Path maestro = repoRoot().resolve("third-party/maestro");
    List<Path> files = new ArrayList<>();
    try (Stream<Path> s = Files.walk(maestro.resolve("maestro-tlaloc/src"))) {
      s.filter(f -> f.toString().endsWith(".java")).forEach(files::add);
    }
    files.add(
        maestro.resolve(
            "maestro-common/src/test/java/com/netflix/maestro/models/stepruntime/"
                + "KubernetesCommandTlalocFieldsTest.java"));
    for (Path f : files) {
      String h = head(f, 12);
      assertTrue(f + " does not carry '" + TLALOC_COPYRIGHT + "'", h.contains(TLALOC_COPYRIGHT));
      assertFalse(f + " carries a Netflix copyright but was written for Tlaloc", h.contains("Netflix, Inc."));
      assertTrue(f + " has no Apache-2.0 header", h.contains("Apache License, Version 2.0"));
    }
  }
}
