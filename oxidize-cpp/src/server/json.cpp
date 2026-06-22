#include "json.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace oxidize {
namespace json {

namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;
  explicit Parser(const std::string& t) : s(t) {}

  [[noreturn]] void fail(const char* msg) {
    throw std::runtime_error(std::string("json: ") + msg + " at offset " +
                             std::to_string(i));
  }
  void skip_ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
                            s[i] == '\r'))
      ++i;
  }
  char peek() {
    skip_ws();
    if (i >= s.size()) fail("unexpected end");
    return s[i];
  }

  Value parse_value() {
    char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Value::string(parse_string());
      case 't': case 'f': return parse_bool();
      case 'n': expect_lit("null"); return Value::null();
      default: return parse_number();
    }
  }

  void expect_lit(const char* lit) {
    for (const char* p = lit; *p; ++p) {
      if (i >= s.size() || s[i] != *p) fail("bad literal");
      ++i;
    }
  }
  Value parse_bool() {
    if (s[i] == 't') { expect_lit("true"); return Value::boolean(true); }
    expect_lit("false");
    return Value::boolean(false);
  }
  Value parse_number() {
    size_t start = i;
    while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '-' ||
                            s[i] == '+' || s[i] == '.' || s[i] == 'e' ||
                            s[i] == 'E'))
      ++i;
    if (i == start) fail("invalid number");
    return Value::number(std::strtod(s.substr(start, i - start).c_str(), nullptr));
  }
  std::string parse_string() {
    if (s[i] != '"') fail("expected string");
    ++i;
    std::string out;
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return out;
      if (c == '\\') {
        if (i >= s.size()) fail("bad escape");
        char e = s[i++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'u': {
            if (i + 4 > s.size()) fail("bad \\u");
            uint32_t cp = std::strtoul(s.substr(i, 4).c_str(), nullptr, 16);
            i += 4;
            // Surrogate pair.
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() &&
                s[i] == '\\' && s[i + 1] == 'u') {
              uint32_t lo = std::strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16);
              i += 6;
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            if (cp < 0x80) {
              out.push_back((char)cp);
            } else if (cp < 0x800) {
              out.push_back((char)(0xC0 | (cp >> 6)));
              out.push_back((char)(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
              out.push_back((char)(0xE0 | (cp >> 12)));
              out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back((char)(0x80 | (cp & 0x3F)));
            } else {
              out.push_back((char)(0xF0 | (cp >> 18)));
              out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
              out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back((char)(0x80 | (cp & 0x3F)));
            }
            break;
          }
          default: out.push_back(e); break;
        }
      } else {
        out.push_back(c);
      }
    }
    fail("unterminated string");
  }
  Value parse_array() {
    ++i;  // [
    Value v = Value::array();
    if (peek() == ']') { ++i; return v; }
    while (true) {
      v.arr->push_back(parse_value());
      char c = peek();
      if (c == ',') { ++i; continue; }
      if (c == ']') { ++i; break; }
      fail("expected , or ]");
    }
    return v;
  }
  Value parse_object() {
    ++i;  // {
    Value v = Value::object();
    if (peek() == '}') { ++i; return v; }
    while (true) {
      if (peek() != '"') fail("expected key");
      std::string key = parse_string();
      if (peek() != ':') fail("expected :");
      ++i;
      v.obj->emplace_back(std::move(key), parse_value());
      char c = peek();
      if (c == ',') { ++i; continue; }
      if (c == '}') { ++i; break; }
      fail("expected , or }");
    }
    return v;
  }
};

}  // namespace

Value parse(const std::string& text) {
  Parser p(text);
  Value v = p.parse_value();
  return v;
}

std::string escape(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back((char)c);
        }
    }
  }
  out.push_back('"');
  return out;
}

std::string dump(const Value& v) {
  switch (v.type) {
    case Value::Type::Null: return "null";
    case Value::Type::Bool: return v.b ? "true" : "false";
    case Value::Type::Number: {
      if (v.num == std::floor(v.num) && std::abs(v.num) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", (long long)v.num);
        return buf;
      }
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g", v.num);
      return buf;
    }
    case Value::Type::String: return escape(v.str);
    case Value::Type::Array: {
      std::string out = "[";
      for (size_t k = 0; k < v.arr->size(); ++k) {
        if (k) out += ",";
        out += dump((*v.arr)[k]);
      }
      return out + "]";
    }
    case Value::Type::Object: {
      std::string out = "{";
      for (size_t k = 0; k < v.obj->size(); ++k) {
        if (k) out += ",";
        out += escape((*v.obj)[k].first) + ":" + dump((*v.obj)[k].second);
      }
      return out + "}";
    }
  }
  return "null";
}

}  // namespace json
}  // namespace oxidize
