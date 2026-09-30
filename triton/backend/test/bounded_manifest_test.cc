// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// The bounded manifest reader against the shared tlaloc-bounded-v1 fixtures
// (harness/bounded-manifest-conformance): every valid manifest reads, every invalid one
// is refused. Built and run by build_backend.sh.
//
//   bounded_manifest_test <fixture directory>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../bounded_mode.h"

namespace fs = std::filesystem;
using triton::backend::tlaloc::BoundedModel;

static std::vector<fs::path>
Fixtures(const fs::path& dir)
{
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() == ".json") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  return files;
}

int
main(int argc, char** argv)
{
  if (argc != 2) {
    std::fprintf(stderr, "usage: bounded_manifest_test <fixture directory>\n");
    return 2;
  }
  int failures = 0, checked = 0;
  for (const char* kind : {"valid", "invalid"}) {
    const bool want_valid = std::string(kind) == "valid";
    const std::vector<fs::path> files = Fixtures(fs::path(argv[1]) / kind);
    if (files.empty()) {
      std::fprintf(stderr, "FAIL no %s fixtures in %s\n", kind, argv[1]);
      ++failures;
    }
    for (const fs::path& f : files) {
      std::ifstream in(f);
      std::stringstream text;
      text << in.rdbuf();
      TRITONSERVER_Error* err = BoundedModel::CheckManifest(text.str());
      const bool read = err == nullptr;
      if (read != want_valid) {
        std::fprintf(
            stderr, "FAIL %s/%s: %s\n", kind, f.filename().c_str(),
            read ? "accepted" : TRITONSERVER_ErrorMessage(err));
        ++failures;
      }
      if (err != nullptr) TRITONSERVER_ErrorDelete(err);
      ++checked;
    }
  }
  std::printf("bounded_manifest_test: %d fixtures, %d failures\n", checked, failures);
  return failures == 0 ? 0 : 1;
}
