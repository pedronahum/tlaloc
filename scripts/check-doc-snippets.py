#!/usr/bin/env python3
"""Compile, and run, the Kotlin snippets the docs mark.

    python3 scripts/check-doc-snippets.py            # every marked snippet
    python3 scripts/check-doc-snippets.py NAME...    # the named ones
    python3 scripts/check-doc-snippets.py --list     # the marked snippets and their files

A snippet is marked by an HTML comment (which GitHub does not render) on the line
before its fence:

    <!-- doc-test: grad-scalar -->

    ```kotlin
    val df = grad { x: Float -> x * x * x }
    ```

`docs/snippets/<name>.kt` supplies what the snippet leaves out: imports, the values
it uses, and the checks of what it computes. Its `// snippet` line is replaced by the
block, indented to match. A snippet without a wrapper must be a whole file. Each file
must define `fun main()`. A wrapper that contains `// compile-only` is compiled and
not run (a snippet that needs a GPU, a checkpoint or a server); one that contains
`// compile-error: <text>` must fail to compile, with <text> in the compiler's output
(a snippet that shows a mistake the compiler refuses), and builds on its own.

The snippets build as one consumer project in build/doc-snippets: the Tlaloc Gradle
plugin, the BOM and the modules from mavenLocal (`./gradlew publishToMavenLocal -x
test` first), each file in its own package. Exit status 0 when every snippet compiles
and every run passes; 1 otherwise, naming the snippet.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPERS = os.path.join(ROOT, "docs", "snippets")
OUT = os.path.join(ROOT, "build", "doc-snippets")

MARKED = re.compile(
    r"^<!-- doc-test: (?P<name>[a-z0-9-]+) -->[ \t]*\n(?:[ \t]*\n)*"
    r"```kotlin[^\n]*\n(?P<body>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
MARKER = re.compile(r"^<!-- doc-test: ", re.MULTILINE)

MODULES = ["core", "autograd", "nn", "stablehlo", "runtime-pjrt", "kptx"]


def doc_files():
    tracked = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).split()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "*.md"], cwd=ROOT, text=True
    ).split()
    return sorted(set(tracked + untracked))


def collect():
    snippets = {}
    problems = []
    for rel in doc_files():
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        found = list(MARKED.finditer(text))
        if len(found) != len(MARKER.findall(text)):
            problems.append(f"{rel}: a doc-test marker is not followed by a ```kotlin block")
        for m in found:
            name = m.group("name")
            line = text.count("\n", 0, m.start()) + 1
            if name in snippets:
                problems.append(f"{rel}:{line}: doc-test {name} is also marked in {snippets[name][0]}")
                continue
            snippets[name] = (f"{rel}:{line}", m.group("body"))
    return snippets, problems


def ident(name):
    return "s_" + name.replace("-", "_")


COMPILE_ERROR = re.compile(r"^// compile-error: (.*)$", re.MULTILINE)


def source(name, body):
    """The snippet's file, and how it is checked: "run", "compile" or ("error", text)."""
    wrapper = os.path.join(WRAPPERS, name + ".kt")
    if not os.path.exists(wrapper):
        return body, "run"
    text = open(wrapper, encoding="utf-8").read()
    out, placed = [], False
    for line in text.splitlines():
        if line.strip() == "// snippet":
            indent = line[: len(line) - len(line.lstrip())]
            out.extend((indent + l) if l.strip() else "" for l in body.rstrip("\n").splitlines())
            placed = True
        else:
            out.append(line)
    if not placed:
        raise SystemExit(f"check-doc-snippets: {wrapper} has no `// snippet` line")
    error = COMPILE_ERROR.search(text)
    mode = ("error", error.group(1).strip()) if error else "compile" if "// compile-only" in text else "run"
    return "\n".join(out) + "\n", mode


def versions():
    build = open(os.path.join(ROOT, "build.gradle.kts"), encoding="utf-8").read()
    toml = open(os.path.join(ROOT, "gradle", "libs.versions.toml"), encoding="utf-8").read()
    return (
        re.search(r'^version = "(.*)"$', build, re.MULTILINE).group(1),
        re.search(r'^kotlin = "(.*)"$', toml, re.MULTILINE).group(1),
    )


def write_build(out):
    version, kotlin = versions()
    src = os.path.join(out, "src", "main", "kotlin")
    if os.path.isdir(src):
        for f in os.listdir(src):
            os.remove(os.path.join(src, f))
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(out, "settings.gradle.kts"), "w") as f:
        f.write(
            'rootProject.name = "tlaloc-doc-snippets"\n'
            "pluginManagement { repositories { mavenLocal(); gradlePluginPortal(); mavenCentral() } }\n"
            "dependencyResolutionManagement { repositories { mavenLocal(); mavenCentral() } }\n"
        )
    deps = "\n".join(f'    implementation("io.github.pedronahum:tlaloc-{m}")' for m in MODULES)
    with open(os.path.join(out, "build.gradle.kts"), "w") as f:
        f.write(
            "plugins {\n"
            f'    kotlin("jvm") version "{kotlin}"\n'
            f'    id("io.github.pedronahum.tlaloc") version "{version}"\n'
            "    application\n"
            "}\n"
            "kotlin {\n    jvmToolchain(25)\n"
            "    compilerOptions { optIn.add(\"io.tlaloc.core.ExperimentalTlalocApi\") }\n}\n"
            "dependencies {\n"
            f'    implementation(platform("io.github.pedronahum:tlaloc-bom:{version}"))\n'
            f"{deps}\n}}\n"
            'application {\n    mainClass.set("doctest.RunKt")\n'
            '    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")\n}\n'
        )
    return src


def write_file(src, name, where, text):
    with open(os.path.join(src, ident(name) + ".kt"), "w") as f:
        f.write(f"// {where}\n@file:Suppress(\"UNUSED_VARIABLE\", \"UNUSED_EXPRESSION\")\n")
        f.write(f"package doctest.{ident(name)}\n\n{text}")


def write_runner(src, runs, compiled):
    with open(os.path.join(src, "Run.kt"), "w") as f:
        f.write("package doctest\n\nfun main() {\n    var failed = 0\n")
        for name, where in runs:
            f.write(
                f'    println("== {name} ({where})")\n'
                f"    try {{ doctest.{ident(name)}.main() }} catch (e: Throwable) {{\n"
                f'        failed++; println("FAIL {name}: $e")\n    }}\n'
            )
        f.write(
            f'    println("doc snippets: {compiled} compiled, {len(runs)} run, $failed failed")\n'
            "    if (failed > 0) kotlin.system.exitProcess(1)\n}\n"
        )


def check_errors(errors):
    """Each snippet that must not compile, in a build of its own."""
    failed = 0
    gradlew = os.path.join(ROOT, "gradlew")
    for name, where, text, expected in errors:
        out = os.path.join(OUT + "-errors", ident(name))
        src = write_build(out)
        write_file(src, name, where, text)
        r = subprocess.run([gradlew, "-p", out, "classes", "--quiet"], cwd=ROOT, capture_output=True, text=True)
        log = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"FAIL {name} ({where}): compiled, but must fail with: {expected}")
            failed += 1
        elif expected not in log:
            print(f"FAIL {name} ({where}): failed to compile without: {expected}\n{log[-3000:]}")
            failed += 1
        else:
            print(f"== {name} ({where}): refused by the compiler, as the doc says")
    return failed


def main(argv):
    snippets, problems = collect()
    if argv[:1] == ["--list"]:
        for name, (where, _) in sorted(snippets.items()):
            wrapped = os.path.exists(os.path.join(WRAPPERS, name + ".kt"))
            print(f"{name:32} {where}{'' if wrapped else '  (whole file)'}")
        return 0
    unused = [
        f[:-3] for f in os.listdir(WRAPPERS) if f.endswith(".kt") and f[:-3] not in snippets
    ] if os.path.isdir(WRAPPERS) else []
    problems += [f"docs/snippets/{n}.kt: no doc marks the snippet {n}" for n in unused]
    chosen = {n: snippets[n] for n in argv} if argv else snippets
    missing = [n for n in argv if n not in snippets]
    problems += [f"no doc marks the snippet {n}" for n in missing]
    if problems:
        print("\n".join(problems))
        return 1
    src = write_build(OUT)
    runs, errors, compiled = [], [], 0
    for name, (where, body) in chosen.items():
        text, mode = source(name, body)
        if isinstance(mode, tuple):
            errors.append((name, where, text, mode[1]))
            continue
        write_file(src, name, where, text)
        compiled += 1
        if mode == "run":
            runs.append((name, where))
    write_runner(src, runs, compiled)
    gradlew = os.path.join(ROOT, "gradlew")
    status = subprocess.call([gradlew, "-p", OUT, "run", "--quiet"], cwd=ROOT)
    failed = check_errors(errors)
    print(f"doc snippets that must not compile: {len(errors)}, {failed} failed")
    return 1 if status or failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
