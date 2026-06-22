#pragma once
// Minimal dependency-free JSON: a recursive-descent parser and a string
// serializer, enough for the OpenAI chat-completions request/response shapes.
// Not a full validator — permissive on input, strict on the subset we emit.

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace oxidize {
namespace json {

struct Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;  // insertion-ordered

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::shared_ptr<Array> arr;
  std::shared_ptr<Object> obj;

  static Value null() { return {}; }
  static Value boolean(bool v) { Value x; x.type = Type::Bool; x.b = v; return x; }
  static Value number(double v) { Value x; x.type = Type::Number; x.num = v; return x; }
  static Value string(std::string v) { Value x; x.type = Type::String; x.str = std::move(v); return x; }
  static Value array() { Value x; x.type = Type::Array; x.arr = std::make_shared<Array>(); return x; }
  static Value object() { Value x; x.type = Type::Object; x.obj = std::make_shared<Object>(); return x; }

  bool is_object() const { return type == Type::Object; }
  bool is_array() const { return type == Type::Array; }
  bool is_string() const { return type == Type::String; }

  // Object field lookup (Null if absent / not an object).
  const Value* find(const std::string& key) const {
    if (type != Type::Object || !obj) return nullptr;
    for (const auto& kv : *obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  std::string get_str(const std::string& key, const std::string& def = "") const {
    const Value* v = find(key);
    return (v && v->type == Type::String) ? v->str : def;
  }
  double get_num(const std::string& key, double def) const {
    const Value* v = find(key);
    return (v && v->type == Type::Number) ? v->num : def;
  }
  bool get_bool(const std::string& key, bool def) const {
    const Value* v = find(key);
    return (v && v->type == Type::Bool) ? v->b : def;
  }
  void set(const std::string& key, Value v) { obj->emplace_back(key, std::move(v)); }
};

// Parse JSON text. Throws std::runtime_error on malformed input.
Value parse(const std::string& text);

// Serialize to compact JSON.
std::string dump(const Value& v);

// Escape a raw string as a JSON string literal (including surrounding quotes).
std::string escape(const std::string& s);

}  // namespace json
}  // namespace oxidize
