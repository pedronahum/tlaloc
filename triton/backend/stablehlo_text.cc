// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo_text.h"

#include <cctype>
#include <regex>
#include <sstream>

namespace tlaloc_triton {

DType
DTypeFromMlir(const std::string& e)
{
  if (e == "f32") return DType::F32;
  if (e == "f64") return DType::F64;
  if (e == "f16") return DType::F16;
  if (e == "bf16") return DType::BF16;
  if (e == "i8" || e == "si8") return DType::I8;
  if (e == "i32" || e == "si32") return DType::I32;
  if (e == "i64" || e == "si64") return DType::I64;
  if (e == "ui8") return DType::U8;
  if (e == "i1") return DType::BOOL;
  return DType::UNSUPPORTED;
}

DType
DTypeFromTriton(const std::string& name)
{
  std::string n = name;
  if (n.rfind("TYPE_", 0) == 0) n = n.substr(5);
  if (n == "FP32") return DType::F32;
  if (n == "FP64") return DType::F64;
  if (n == "FP16") return DType::F16;
  if (n == "BF16") return DType::BF16;
  if (n == "INT8") return DType::I8;
  if (n == "INT32") return DType::I32;
  if (n == "INT64") return DType::I64;
  if (n == "UINT8") return DType::U8;
  if (n == "BOOL") return DType::BOOL;
  return DType::UNSUPPORTED;
}

const char*
TritonName(DType t)
{
  switch (t) {
    case DType::F32: return "FP32";
    case DType::F64: return "FP64";
    case DType::F16: return "FP16";
    case DType::BF16: return "BF16";
    case DType::I8: return "INT8";
    case DType::I32: return "INT32";
    case DType::I64: return "INT64";
    case DType::U8: return "UINT8";
    case DType::BOOL: return "BOOL";
    default: return "UNSUPPORTED";
  }
}

size_t
ByteWidth(DType t)
{
  switch (t) {
    case DType::F64:
    case DType::I64: return 8;
    case DType::F32:
    case DType::I32: return 4;
    case DType::F16:
    case DType::BF16: return 2;
    case DType::I8:
    case DType::U8:
    case DType::BOOL: return 1;
    default: return 0;
  }
}

std::string
ShapeString(const std::vector<int64_t>& dims)
{
  std::ostringstream s;
  s << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i) s << ",";
    s << dims[i];
  }
  s << "]";
  return s.str();
}

bool
IsMlirBytecode(const std::string& bytes)
{
  return bytes.size() >= 4 && bytes[0] == 'M' && bytes[1] == 'L' &&
         static_cast<unsigned char>(bytes[2]) == 0xEF && bytes[3] == 'R';
}

namespace {

bool
IsIdChar(char c)
{
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' ||
         c == '.';
}

// Blanks out `//` comments (outside double-quoted strings) so that nothing in
// a comment is read as code. Offsets are preserved.
std::string
StripComments(const std::string& text)
{
  std::string out = text;
  bool in_string = false;
  for (size_t i = 0; i < out.size(); ++i) {
    char c = out[i];
    if (c == '\n') {
      in_string = false;
      continue;
    }
    if (in_string) {
      if (c == '\\' && i + 1 < out.size()) {
        ++i;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '/' && i + 1 < out.size() && out[i + 1] == '/') {
      while (i < out.size() && out[i] != '\n') out[i++] = ' ';
      --i;
    }
  }
  return out;
}

void
SkipSpace(const std::string& s, size_t* i)
{
  while (*i < s.size() && std::isspace(static_cast<unsigned char>(s[*i]))) ++*i;
}

// Given s[open] is one of ( [ { <, returns the index of its matching closer,
// or npos. "->" is not a closer.
size_t
MatchClose(const std::string& s, size_t open)
{
  int depth = 0;
  bool in_string = false;
  for (size_t i = open; i < s.size(); ++i) {
    char c = s[i];
    if (in_string) {
      if (c == '\\') {
        ++i;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '(' || c == '[' || c == '{' || c == '<') {
      ++depth;
    } else if (c == ')' || c == ']' || c == '}' || (c == '>' && s[i - 1] != '-')) {
      if (--depth == 0) return i;
    }
  }
  return std::string::npos;
}

// Splits s[begin, end) at commas that are not nested in brackets or strings.
std::vector<std::string>
SplitTopLevel(const std::string& s, size_t begin, size_t end)
{
  std::vector<std::string> parts;
  int depth = 0;
  bool in_string = false;
  size_t start = begin;
  for (size_t i = begin; i < end; ++i) {
    char c = s[i];
    if (in_string) {
      if (c == '\\') {
        ++i;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '(' || c == '[' || c == '{' || c == '<') {
      ++depth;
    } else if (c == ')' || c == ']' || c == '}' || (c == '>' && s[i - 1] != '-')) {
      --depth;
    } else if (c == ',' && depth == 0) {
      parts.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  std::string last = s.substr(start, end - start);
  if (last.find_first_not_of(" \t\r\n") != std::string::npos) parts.push_back(last);
  return parts;
}

// Finds the first "tensor<...>" in `item` and parses it.
bool
TensorIn(
    const std::string& item, const std::string& what, TensorType* out,
    std::string* error)
{
  size_t p = item.find("tensor<");
  if (p == std::string::npos) {
    *error = what + " is not a tensor: '" + item + "'";
    return false;
  }
  size_t close = MatchClose(item, p + 6);
  if (close == std::string::npos) {
    *error = what + " has an unterminated tensor type";
    return false;
  }
  std::string perr;
  if (!ParseTensorType(item.substr(p, close - p + 1), out, &perr)) {
    *error = what + ": " + perr;
    return false;
  }
  return true;
}

std::string
ReplaceSymbol(const std::string& text, const std::string& from, const std::string& to)
{
  std::string needle = "@" + from;
  std::string out;
  size_t pos = 0;
  while (true) {
    size_t hit = text.find(needle, pos);
    if (hit == std::string::npos) break;
    size_t after = hit + needle.size();
    bool whole = after >= text.size() || !IsIdChar(text[after]);
    out.append(text, pos, hit - pos);
    out.append(whole ? "@" + to : needle);
    pos = after;
  }
  out.append(text, pos, std::string::npos);
  return out;
}

}  // namespace

bool
ParseTensorType(const std::string& text, TensorType* out, std::string* error)
{
  const std::string prefix = "tensor<";
  if (text.rfind(prefix, 0) != 0 || text.back() != '>') {
    *error = "'" + text + "' is not a tensor type";
    return false;
  }
  std::string body = text.substr(prefix.size(), text.size() - prefix.size() - 1);
  // Drop an encoding (", #enc") if there is one.
  std::vector<std::string> enc = SplitTopLevel(body, 0, body.size());
  if (!enc.empty()) body = enc[0];
  if (!body.empty() && body[0] == '*') {
    *error = "'" + text + "' is unranked; only ranked tensors are served";
    return false;
  }
  TensorType t;
  t.text = text;
  size_t start = 0;
  while (true) {
    size_t x = body.find('x', start);
    std::string part = body.substr(start, x == std::string::npos ? std::string::npos : x - start);
    bool is_dim = x != std::string::npos &&
                  (part == "?" || (!part.empty() &&
                                   part.find_first_not_of("0123456789") == std::string::npos));
    if (!is_dim) {
      t.element = body.substr(start);
      break;
    }
    t.dims.push_back(part == "?" ? -1 : std::stoll(part));
    start = x + 1;
  }
  while (!t.element.empty() && std::isspace(static_cast<unsigned char>(t.element.back()))) {
    t.element.pop_back();
  }
  t.dtype = DTypeFromMlir(t.element);
  *out = t;
  return true;
}

bool
ListFunctions(
    const std::string& module_text, std::vector<FunctionSignature>* out,
    std::string* error)
{
  const std::string s = StripComments(module_text);
  const std::string kw = "func.func";
  size_t pos = 0;
  while (true) {
    size_t hit = s.find(kw, pos);
    if (hit == std::string::npos) break;
    size_t i = hit + kw.size();
    pos = i;
    if (hit > 0 && IsIdChar(s[hit - 1])) continue;
    if (i < s.size() && IsIdChar(s[i])) continue;
    SkipSpace(s, &i);
    for (const char* vis : {"public", "private", "nested"}) {
      std::string v(vis);
      if (s.compare(i, v.size(), v) == 0 && i + v.size() < s.size() &&
          std::isspace(static_cast<unsigned char>(s[i + v.size()]))) {
        i += v.size();
        SkipSpace(s, &i);
        break;
      }
    }
    if (i >= s.size() || s[i] != '@') {
      *error = "a func.func without a symbol name";
      return false;
    }
    ++i;
    FunctionSignature f;
    if (i < s.size() && s[i] == '"') {
      size_t q = s.find('"', i + 1);
      if (q == std::string::npos) {
        *error = "unterminated quoted function name";
        return false;
      }
      f.name = s.substr(i + 1, q - i - 1);
      i = q + 1;
    } else {
      size_t b = i;
      while (i < s.size() && IsIdChar(s[i])) ++i;
      f.name = s.substr(b, i - b);
    }
    SkipSpace(s, &i);
    if (i >= s.size() || s[i] != '(') {
      *error = "@" + f.name + ": expected '(' after the function name";
      return false;
    }
    size_t close = MatchClose(s, i);
    if (close == std::string::npos) {
      *error = "@" + f.name + ": unterminated argument list";
      return false;
    }
    std::vector<std::string> args = SplitTopLevel(s, i + 1, close);
    for (size_t a = 0; a < args.size(); ++a) {
      TensorType t;
      if (!TensorIn(args[a], "@" + f.name + " argument " + std::to_string(a), &t, error)) {
        return false;
      }
      f.args.push_back(t);
    }
    i = close + 1;
    SkipSpace(s, &i);
    if (s.compare(i, 2, "->") == 0) {
      i += 2;
      SkipSpace(s, &i);
      std::vector<std::string> results;
      if (i < s.size() && s[i] == '(') {
        size_t rclose = MatchClose(s, i);
        if (rclose == std::string::npos) {
          *error = "@" + f.name + ": unterminated result list";
          return false;
        }
        results = SplitTopLevel(s, i + 1, rclose);
        i = rclose + 1;
      } else {
        size_t tclose = s.compare(i, 7, "tensor<") == 0 ? MatchClose(s, i + 6)
                                                         : std::string::npos;
        if (tclose == std::string::npos) {
          *error = "@" + f.name + ": the result is not a tensor type";
          return false;
        }
        results.push_back(s.substr(i, tclose - i + 1));
        i = tclose + 1;
      }
      for (size_t r = 0; r < results.size(); ++r) {
        TensorType t;
        if (!TensorIn(results[r], "@" + f.name + " result " + std::to_string(r), &t, error)) {
          return false;
        }
        f.results.push_back(t);
      }
    }
    pos = i;
    out->push_back(f);
  }
  return true;
}

bool
SelectEntry(
    const std::vector<FunctionSignature>& functions, const std::string& entry,
    FunctionSignature* out, std::string* error)
{
  if (functions.empty()) {
    *error = "the artifact contains no func.func";
    return false;
  }
  const std::string want = entry.empty() ? "main" : entry;
  for (const auto& f : functions) {
    if (f.name == want) {
      *out = f;
      return true;
    }
  }
  if (entry.empty()) {
    *out = functions.front();
    return true;
  }
  std::string names;
  for (const auto& f : functions) names += (names.empty() ? "@" : ", @") + f.name;
  *error = "entry function @" + entry + " is not in the artifact; it defines " + names;
  return false;
}

std::string
PrepareForXla(const std::string& module_text, const std::string& entry)
{
  std::string text = module_text;
  if (entry != "main") {
    // A different function already called main would collide; move it aside.
    text = ReplaceSymbol(text, "main", "tlaloc_renamed_main");
    text = ReplaceSymbol(text, entry, "main");
  }
  text = std::regex_replace(
      text, std::regex(R"(func\.func\s+private\s+@main\b)"), "func.func @main");

  // Wrap in a module unless the first code line already opens one.
  const std::string stripped = StripComments(text);
  size_t first = stripped.find_first_not_of(" \t\r\n");
  bool has_module = first != std::string::npos &&
                    stripped.compare(first, 6, "module") == 0 &&
                    (first + 6 >= stripped.size() || !IsIdChar(stripped[first + 6]));
  if (!has_module) {
    text = "module @tlaloc_triton {\n" + text + (text.empty() || text.back() == '\n' ? "" : "\n") +
           "}\n";
  }
  return text;
}

}  // namespace tlaloc_triton
