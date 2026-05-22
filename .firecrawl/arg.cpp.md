#include "arg.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "download.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "preset.h"

// fix problem with std::min and std::max
#if defined(\_WIN32)
#define WIN32\_LEAN\_AND\_MEAN
#ifndef NOMINMAX
\# define NOMINMAX
#endif
#include
#endif

#define JSON\_ASSERT GGML\_ASSERT
#include

#include
#include
#include
#include
#include
#include
#include
#include
#include
#include  // for hardware\_concurrency
#include

#ifndef \_\_EMSCRIPTEN\_\_
#ifdef \_\_linux\_\_
#include
#elif defined(\_WIN32)
\# if !defined(PATH\_MAX)
\# define PATH\_MAX MAX\_PATH
\# endif
#elif defined(\_AIX)
#include
#else
#include
#endif
#endif

#define LLAMA\_MAX\_URL\_LENGTH 2084 // Maximum URL Length in Chrome: 2083

extern const char \* LICENSES\[\];

using json = nlohmann::ordered\_json;
using namespace common\_arg\_utils;

static std::initializer\_list mmproj\_examples = {
 LLAMA\_EXAMPLE\_MTMD,
 LLAMA\_EXAMPLE\_SERVER,
 LLAMA\_EXAMPLE\_CLI,
};

static std::string read\_file(const std::string & fname) {
 std::ifstream file(fname);
 if (!file) {
 throw std::runtime\_error(string\_format("error: failed to open file '%s'\\n", fname.c\_str()));
 }
 std::string content((std::istreambuf\_iterator(file)), std::istreambuf\_iterator());
 file.close();
 return content;
}

static const std::vector & get\_common\_arg\_defs() {
 static const std::vector options = \[\] {
 common\_params params;
 auto ctx = common\_params\_parser\_init(params, LLAMA\_EXAMPLE\_SERVER, nullptr);
 return ctx.options;
 }();
 return options;
}

common\_arg & common\_arg::set\_examples(std::initializer\_list examples) {
 this->examples = examples;
 return \*this;
}

common\_arg & common\_arg::set\_excludes(std::initializer\_list excludes) {
 this->excludes = excludes;
 return \*this;
}

common\_arg & common\_arg::set\_env(const char \* env) {
 help = help + "\\n(env: " + env + ")";
 this->env = env;
 return \*this;
}

common\_arg & common\_arg::set\_sampling() {
 is\_sampling = true;
 return \*this;
}

common\_arg & common\_arg::set\_spec() {
 is\_spec = true;
 return \*this;
}

common\_arg & common\_arg::set\_preset\_only() {
 is\_preset\_only = true;
 return \*this;
}

bool common\_arg::in\_example(enum llama\_example ex) {
 return examples.find(ex) != examples.end();
}

bool common\_arg::is\_exclude(enum llama\_example ex) {
 return excludes.find(ex) != excludes.end();
}

bool common\_arg::get\_value\_from\_env(std::string & output) const {
 if (env == nullptr) return false;
 if (!args\_neg.empty()) {
 // for compatibility, we need to check LLAMA\_ARG\_NO\_ env as well
 std::string neg\_env = env;
 string\_replace\_all(neg\_env, "LLAMA\_ARG\_", "LLAMA\_ARG\_NO\_");
 char \* neg\_value = std::getenv(neg\_env.c\_str());
 if (neg\_value) {
 output = "0"; // falsey
 return true;
 }
 }
 char \* value = std::getenv(env);
 if (value) {
 output = value;
 return true;
 }
 return false;
}

bool common\_arg::has\_value\_from\_env() const {
 if (env != nullptr && !args\_neg.empty()) {
 // for compatibility, we need to check LLAMA\_ARG\_NO\_ env as well
 std::string neg\_env = env;
 string\_replace\_all(neg\_env, "LLAMA\_ARG\_", "LLAMA\_ARG\_NO\_");
 if (std::getenv(neg\_env.c\_str())) {
 return true;
 }
 }
 return env != nullptr && std::getenv(env);
}

static std::vector break\_str\_into\_lines(std::string input, size\_t max\_char\_per\_line) {
 std::vector result;
 std::istringstream iss(input);
 std::string line;
 auto add\_line = \[&\](const std::string& l) {
 if (l.length() <= max\_char\_per\_line) {
 result.push\_back(l);
 } else {
 std::istringstream line\_stream(l);
 std::string word, current\_line;
 while (line\_stream >> word) {
 if (current\_line.length() + !current\_line.empty() + word.length() > max\_char\_per\_line) {
 if (!current\_line.empty()) result.push\_back(current\_line);
 current\_line = word;
 } else {
 current\_line += (!current\_line.empty() ? " " : "") + word;
 }
 }
 if (!current\_line.empty()) result.push\_back(current\_line);
 }
 };
 while (std::getline(iss, line)) {
 add\_line(line);
 }
 return result;
}

std::string common\_arg::to\_string() const {
 // params for printing to console
 const static int n\_leading\_spaces = 40;
 const static int n\_char\_per\_line\_help = 70; // TODO: detect this based on current console
 std::string leading\_spaces(n\_leading\_spaces, ' ');

 std::ostringstream ss;
 auto all\_args = get\_args(); // also contains args\_neg
 for (const auto & arg : all\_args) {
 if (arg == all\_args.front()) {
 if (all\_args.size() == 1) {
 ss << arg;
 } else {
 // first arg is usually abbreviation, we need padding to make it more beautiful
 auto tmp = std::string(arg) + ", ";
 auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
 ss << tmp << spaces;
 }
 } else {
 ss << arg << (arg != all\_args.back() ? ", " : "");
 }
 }
 if (value\_hint) ss << " " << value\_hint;
 if (value\_hint\_2) ss << " " << value\_hint\_2;
 if (ss.tellp() > n\_leading\_spaces - 3) {
 // current line is too long, add new line
 ss << "\\n" << leading\_spaces;
 } else {
 // padding between arg and help, same line
 ss << std::string(leading\_spaces.size() - ss.tellp(), ' ');
 }
 const auto help\_lines = break\_str\_into\_lines(help, n\_char\_per\_line\_help);
 for (const auto & line : help\_lines) {
 ss << (&line == &help\_lines.front() ? "" : leading\_spaces) << line << "\\n";
 }
 return ss.str();
}

std::vector common\_arg::get\_args() const {
 std::vector result;
 for (const auto & arg : args) {
 result.push\_back(std::string(arg));
 }
 for (const auto & arg : args\_neg) {
 result.push\_back(std::string(arg));
 }
 return result;
}

std::vector common\_arg::get\_env() const {
 std::vector result;
 if (env) {
 result.push\_back(std::string(env));
 }
 if (!args\_neg.empty() && env) {
 // for compatibility, we need to add LLAMA\_ARG\_NO\_ variant
 std::string neg\_env = env;
 string\_replace\_all(neg\_env, "LLAMA\_ARG\_", "LLAMA\_ARG\_NO\_");
 result.push\_back(neg\_env);
 }
 return result;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void parse\_tensor\_buffer\_overrides(const std::string & value, std::vector & overrides) {
 ggml\_backend\_load\_all();

 std::map buft\_list;
 for (size\_t i = 0; i < ggml\_backend\_dev\_count(); ++i) {
 auto \* dev = ggml\_backend\_dev\_get(i);
 auto \* buft = ggml\_backend\_dev\_buffer\_type(dev);
 if (buft) {
 buft\_list\[ggml\_backend\_buft\_name(buft)\] = buft;
 }
 }

 for (const auto & override : string\_split(value, ',')) {
 std::string::size\_type pos = override.find('=');
 if (pos == std::string::npos) {
 throw std::invalid\_argument("invalid value");
 }
 std::string tensor\_name = override.substr(0, pos);
 std::string buffer\_type = override.substr(pos + 1);

 if (buft\_list.find(buffer\_type) == buft\_list.end()) {
 printf("Available buffer types:\\n");
 for (const auto & it : buft\_list) {
 printf(" %s\\n", ggml\_backend\_buft\_name(it.second));
 }
 throw std::invalid\_argument("unknown buffer type");
 }
 // keep strings alive and avoid leaking memory by storing them in a static vector
 static std::list buft\_overrides;
 buft\_overrides.push\_back(tensor\_name);
 overrides.push\_back({buft\_overrides.back().c\_str(), buft\_list.at(buffer\_type)});
 }
}

static std::string clean\_file\_name(const std::string & fname) {
 std::string clean\_fname = fname;
 string\_replace\_all(clean\_fname, "\\\", "\_");
 string\_replace\_all(clean\_fname, "/", "\_");
 return clean\_fname;
}

static bool common\_params\_handle\_remote\_preset(common\_params & params, llama\_example ex) {
 GGML\_ASSERT(!params.model.hf\_repo.empty());

 // the returned hf\_repo is without tag
 auto \[hf\_repo, hf\_tag\] = common\_download\_split\_repo\_tag(params.model.hf\_repo);

 // "latest" tag (default if not specified) is translated to "default" preset
 if (hf\_tag == "latest") {
 hf\_tag = "default";
 }

 std::string model\_endpoint = common\_get\_model\_endpoint();
 auto preset\_url = model\_endpoint + hf\_repo + "/resolve/main/preset.ini";

 // prepare local path for caching
 auto preset\_fname = clean\_file\_name(hf\_repo + "\_preset.ini");
 auto preset\_path = fs\_get\_cache\_file(preset\_fname);
 common\_download\_opts opts;
 opts.bearer\_token = params.hf\_token;
 opts.offline = params.offline;

 LOG\_TRC("%s: looking for remote preset at %s\\n", \_\_func\_\_, preset\_url.c\_str());
 const int status = common\_download\_file\_single(preset\_url, preset\_path, opts);
 const bool has\_preset = status >= 200 && status < 400;

 // remote preset is optional, so we don't error out if not found
 if (has\_preset) {
 LOG\_TRC("%s: applying remote preset from %s\\n", \_\_func\_\_, preset\_url.c\_str());
 common\_preset\_context ctx(ex, /\* only\_remote\_allowed \*/ true);
 common\_preset global;
 auto remote\_presets = ctx.load\_from\_ini(preset\_path, global);
 remote\_presets = ctx.cascade(global, remote\_presets);
 if (remote\_presets.find(hf\_tag) != remote\_presets.end()) {
 common\_preset preset = remote\_presets.at(hf\_tag);
 LOG\_INF("\\n%s", preset.to\_ini().c\_str()); // to\_ini already added trailing newline
 preset.apply\_to\_params(params);
 } else {
 throw std::runtime\_error("Remote preset.ini does not contain \[" + std::string(hf\_tag) + "\] section");
 }
 } else {
 LOG\_TRC("%s: no remote preset found, skipping\\n", \_\_func\_\_);
 }

 return has\_preset;
}

struct handle\_model\_result {
 bool found\_mmproj = false;
 common\_params\_model mmproj;

 bool found\_mtp = false;
 common\_params\_model mtp;
};

static handle\_model\_result common\_params\_handle\_model(struct common\_params\_model & model,
 const std::string & bearer\_token,
 bool offline,
 bool search\_mtp = false) {
 handle\_model\_result result;

 if (!model.docker\_repo.empty()) {
 model.path = common\_docker\_resolve\_model(model.docker\_repo);
 model.name = model.docker\_repo;
 } else if (!model.hf\_repo.empty()) {
 // If -m was used with -hf, treat the model "path" as the hf\_file to download
 if (model.hf\_file.empty() && !model.path.empty()) {
 model.hf\_file = model.path;
 model.path = "";
 }
 common\_download\_opts opts;
 opts.bearer\_token = bearer\_token;
 opts.offline = offline;
 auto download\_result = common\_download\_model(model, opts, true, search\_mtp);

 if (download\_result.model\_path.empty()) {
 throw std::runtime\_error("failed to download model from Hugging Face");
 }

 model.name = model.hf\_repo;
 model.path = download\_result.model\_path;

 if (!download\_result.mmproj\_path.empty()) {
 result.found\_mmproj = true;
 result.mmproj.path = download\_result.mmproj\_path;
 }

 if (!download\_result.mtp\_path.empty()) {
 result.found\_mtp = true;
 result.mtp.path = download\_result.mtp\_path;
 }
 } else if (!model.url.empty()) {
 if (model.path.empty()) {
 auto f = string\_split(model.url, '#').front();
 f = string\_split(f, '?').front();
 model.path = fs\_get\_cache\_file(string\_split(f, '/').back());
 }

 common\_download\_opts opts;
 opts.bearer\_token = bearer\_token;
 opts.offline = offline;
 auto download\_result = common\_download\_model(model, opts);
 if (download\_result.model\_path.empty()) {
 throw std::runtime\_error("failed to download model from " + model.url);
 }
 }

 return result;
}

const std::vector kv\_cache\_types = {
 GGML\_TYPE\_F32,
 GGML\_TYPE\_F16,
 GGML\_TYPE\_BF16,
 GGML\_TYPE\_Q8\_0,
 GGML\_TYPE\_Q4\_0,
 GGML\_TYPE\_Q4\_1,
 GGML\_TYPE\_IQ4\_NL,
 GGML\_TYPE\_Q5\_0,
 GGML\_TYPE\_Q5\_1,
};

static ggml\_type kv\_cache\_type\_from\_str(const std::string & s) {
 for (const auto & type : kv\_cache\_types) {
 if (ggml\_type\_name(type) == s) {
 return type;
 }
 }
 throw std::runtime\_error("Unsupported cache type: " + s);
}

static std::string get\_all\_kv\_cache\_types() {
 std::ostringstream msg;
 for (const auto & type : kv\_cache\_types) {
 msg << ggml\_type\_name(type) << (&type == &kv\_cache\_types.back() ? "" : ", ");
 }
 return msg.str();
}

static bool parse\_bool\_value(const std::string & value) {
 if (is\_truthy(value)) {
 return true;
 } else if (is\_falsey(value)) {
 return false;
 } else {
 throw std::invalid\_argument("invalid boolean value");
 }
}

\[\[noreturn\]\] static void arg\_removed(const std::string & msg) {
 throw std::invalid\_argument("the argument has been removed. " + msg);
}

//
// CLI argument parsing functions
//

void common\_params\_handle\_models(common\_params & params, llama\_example curr\_ex) {
 const bool spec\_type\_draft\_mtp = std::find(params.speculative.types.begin(),
 params.speculative.types.end(),
 COMMON\_SPECULATIVE\_TYPE\_DRAFT\_MTP) != params.speculative.types.end();

 auto res = common\_params\_handle\_model(params.model, params.hf\_token, params.offline, spec\_type\_draft\_mtp);
 if (params.no\_mmproj) {
 params.mmproj = {};
 } else if (res.found\_mmproj && params.mmproj.path.empty() && params.mmproj.url.empty()) {
 // optionally, handle mmproj model when -hf is specified
 params.mmproj = res.mmproj;
 }
 // only download mmproj if the current example is using it
 for (const auto & ex : mmproj\_examples) {
 if (curr\_ex == ex) {
 common\_params\_handle\_model(params.mmproj, params.hf\_token, params.offline);
 break;
 }
 }
 // when --spec-type mtp is set and no draft model was provided explicitly,
 // fall back to the MTP head discovered alongside the -hf model
 if (spec\_type\_draft\_mtp && res.found\_mtp &&
 params.speculative.draft.mparams.path.empty() &&
 params.speculative.draft.mparams.hf\_repo.empty() &&
 params.speculative.draft.mparams.url.empty()) {
 params.speculative.draft.mparams.path = res.mtp.path;
 }
 common\_params\_handle\_model(params.speculative.draft.mparams, params.hf\_token, params.offline);
 common\_params\_handle\_model(params.vocoder.model, params.hf\_token, params.offline);
}

static bool common\_params\_parse\_ex(int argc, char \*\* argv, common\_params\_context & ctx\_arg) {
 common\_params & params = ctx\_arg.params;

 // setup log directly from params.verbosity: see tools/cli/cli.cpp
 common\_log\_set\_verbosity\_thold(params.verbosity);

 std::unordered\_map\> arg\_to\_options;
 for (auto & opt : ctx\_arg.options) {
 for (const auto & arg : opt.args) {
 arg\_to\_options\[arg\] = {&opt, /\* is\_positive \*/ true};
 }
 for (const auto & arg : opt.args\_neg) {
 arg\_to\_options\[arg\] = {&opt, /\* is\_positive \*/ false};
 }
 }

 // handle environment variables
 for (auto & opt : ctx\_arg.options) {
 std::string value;
 if (opt.get\_value\_from\_env(value)) {
 try {
 if (opt.handler\_void && is\_truthy(value)) {
 opt.handler\_void(params);
 }
 if (opt.handler\_int) {
 opt.handler\_int(params, std::stoi(value));
 }
 if (opt.handler\_bool) {
 opt.handler\_bool(params, parse\_bool\_value(value));
 }
 if (opt.handler\_string) {
 opt.handler\_string(params, value);
 continue;
 }
 } catch (std::exception & e) {
 throw std::invalid\_argument(string\_format(
 "error while handling environment variable \\"%s\\": %s\\n\\n", opt.env, e.what()));
 }
 }
 }

 // handle command line arguments
 auto check\_arg = \[&\](int i) {
 if (i+1 >= argc) {
 throw std::invalid\_argument("expected value for argument");
 }
 };

 auto parse\_cli\_args = \[&\]() {
 std::set seen\_args;

 for (int i = 1; i < argc; i++) {
 const std::string arg\_prefix = "--";

 std::string arg = argv\[i\];
 if (arg.compare(0, arg\_prefix.size(), arg\_prefix) == 0) {
 std::replace(arg.begin(), arg.end(), '\_', '-');
 }
 if (arg\_to\_options.find(arg) == arg\_to\_options.end()) {
 throw std::invalid\_argument(string\_format("error: invalid argument: %s", arg.c\_str()));
 }
 if (!seen\_args.insert(arg).second) {
 const bool skip = (arg == "--spec-type");

 if (!skip) {
 LOG\_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\\n", arg.c\_str());
 }
 }
 auto & tmp = arg\_to\_options\[arg\];
 auto opt = \*tmp.first;
 bool is\_positive = tmp.second;
 if (opt.has\_value\_from\_env()) {
 fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\\n", opt.env, arg.c\_str());
 }
 try {
 if (opt.handler\_void) {
 opt.handler\_void(params);
 continue;
 }
 if (opt.handler\_bool) {
 opt.handler\_bool(params, is\_positive);
 continue;
 }

 // arg with single value
 check\_arg(i);
 std::string val = argv\[++i\];
 if (opt.handler\_int) {
 opt.handler\_int(params, std::stoi(val));
 continue;
 }
 if (opt.handler\_string) {
 opt.handler\_string(params, val);
 continue;
 }

 // arg with 2 values
 check\_arg(i);
 std::string val2 = argv\[++i\];
 if (opt.handler\_str\_str) {
 opt.handler\_str\_str(params, val, val2);
 continue;
 }
 } catch (std::exception & e) {
 throw std::invalid\_argument(string\_format(
 "error while handling argument \\"%s\\": %s\\n\\n"
 "usage:\\n%s\\n\\nto show complete usage, run with -h",
 arg.c\_str(), e.what(), opt.to\_string().c\_str()));
 }
 }
 };

 // parse the first time to get -hf option (used for remote preset)
 parse\_cli\_args();

 // export\_graph\_ops loads only metadata
 const bool skip\_model\_download = ctx\_arg.ex == LLAMA\_EXAMPLE\_EXPORT\_GRAPH\_OPS;

 // maybe handle remote preset
 if (!params.model.hf\_repo.empty() && !skip\_model\_download) {
 std::string cli\_hf\_repo = params.model.hf\_repo;
 bool has\_preset = common\_params\_handle\_remote\_preset(params, ctx\_arg.ex);

 // special case: if hf\_repo explicitly set by preset, we need to preserve it (ignore CLI value)
 // this is useful when we have one HF repo pointing to other HF repos (one model - multiple GGUFs)
 std::string preset\_hf\_repo = params.model.hf\_repo;
 bool preset\_has\_hf\_repo = preset\_hf\_repo != cli\_hf\_repo;

 if (has\_preset) {
 // re-parse CLI args to override preset values
 parse\_cli\_args();
 }

 // preserve hf\_repo from preset if needed
 if (preset\_has\_hf\_repo) {
 params.model.hf\_repo = preset\_hf\_repo;
 }
 }

 postprocess\_cpu\_params(params.cpuparams, nullptr);
 postprocess\_cpu\_params(params.cpuparams\_batch, ¶ms.cpuparams);

 postprocess\_cpu\_params(params.speculative.draft.cpuparams, ¶ms.cpuparams);
 postprocess\_cpu\_params(params.speculative.draft.cpuparams\_batch, ¶ms.cpuparams\_batch);

 if (params.prompt\_cache\_all && (params.interactive \|\| params.interactive\_first)) {
 throw std::invalid\_argument("error: --prompt-cache-all not supported in interactive mode yet\\n");
 }

 // handle model and download
 if (!skip\_model\_download) {
 common\_params\_handle\_models(params, ctx\_arg.ex);
 }

 // model is required (except for server)
 // TODO @ngxson : maybe show a list of available models in CLI in this case
 if (params.model.path.empty() && ctx\_arg.ex != LLAMA\_EXAMPLE\_SERVER && !skip\_model\_download && !params.usage && !params.completion) {
 throw std::invalid\_argument("error: --model is required\\n");
 }

 if (params.escape) {
 string\_process\_escapes(params.prompt);
 string\_process\_escapes(params.input\_prefix);
 string\_process\_escapes(params.input\_suffix);
 for (auto & antiprompt : params.antiprompt) {
 string\_process\_escapes(antiprompt);
 }
 for (auto & seq\_breaker : params.sampling.dry\_sequence\_breakers) {
 string\_process\_escapes(seq\_breaker);
 }
 }

 if (!params.kv\_overrides.empty()) {
 params.kv\_overrides.emplace\_back();
 params.kv\_overrides.back().key\[0\] = 0;
 }

 // pad tensor\_buft\_overrides for llama\_params\_fit:
 const size\_t ntbo = llama\_max\_tensor\_buft\_overrides();
 while (params.tensor\_buft\_overrides.size() < ntbo) {
 params.tensor\_buft\_overrides.push\_back({nullptr, nullptr});
 }

 if (!params.speculative.draft.tensor\_buft\_overrides.empty()) {
 params.speculative.draft.tensor\_buft\_overrides.push\_back({nullptr, nullptr});
 }

 if (!params.chat\_template.empty() && !common\_chat\_verify\_template(params.chat\_template, params.use\_jinja)) {
 throw std::runtime\_error(string\_format(
 "error: the supplied chat template is not supported: %s%s\\n",
 params.chat\_template.c\_str(),
 params.use\_jinja ? "" : "\\nnote: llama.cpp was started without --jinja, we only support commonly used templates"
 ));
 }

 return true;
}

static void common\_params\_print\_usage(common\_params\_context & ctx\_arg) {
 auto print\_options = \[\](std::vector & options) {
 for (common\_arg \* opt : options) {
 printf("%s", opt->to\_string().c\_str());
 }
 };

 std::vector common\_options;
 std::vector sampling\_options;
 std::vector spec\_options;
 std::vector specific\_options;
 for (auto & opt : ctx\_arg.options) {
 // in case multiple LLAMA\_EXAMPLE\_\* are set, we prioritize the LLAMA\_EXAMPLE\_\* matching current example
 if (opt.is\_sampling) {
 sampling\_options.push\_back(&opt);
 } else if (opt.is\_spec) {
 spec\_options.push\_back(&opt);
 } else if (opt.in\_example(ctx\_arg.ex)) {
 specific\_options.push\_back(&opt);
 } else {
 common\_options.push\_back(&opt);
 }
 }
 printf("----- common params -----\\n\\n");
 print\_options(common\_options);
 printf("\\n\\n----- sampling params -----\\n\\n");
 print\_options(sampling\_options);
 printf("\\n\\n----- speculative params -----\\n\\n");
 print\_options(spec\_options);
 // TODO: maybe convert enum llama\_example to string
 printf("\\n\\n----- example-specific params -----\\n\\n");
 print\_options(specific\_options);
}

static void common\_params\_print\_completion(common\_params\_context & ctx\_arg) {
 std::vector common\_options;
 std::vector sampling\_options;
 std::vector spec\_options;
 std::vector specific\_options;

 for (auto & opt : ctx\_arg.options) {
 if (opt.is\_sampling) {
 sampling\_options.push\_back(&opt);
 } else if (opt.is\_spec) {
 spec\_options.push\_back(&opt);
 } else if (opt.in\_example(ctx\_arg.ex)) {
 specific\_options.push\_back(&opt);
 } else {
 common\_options.push\_back(&opt);
 }
 }

 printf("\_llama\_completions() {\\n");
 printf(" local cur prev opts\\n");
 printf(" COMPREPLY=()\\n");
 printf(" cur=\\"${COMP\_WORDS\[COMP\_CWORD\]}\\"\\n");
 printf(" prev=\\"${COMP\_WORDS\[COMP\_CWORD-1\]}\\"\\n\\n");

 printf(" opts=\\"");
 auto print\_options = \[\](const std::vector & options) {
 for (const common\_arg \* opt : options) {
 for (const char \* arg : opt->args) {
 printf("%s ", arg);
 }
 }
 };

 print\_options(common\_options);
 print\_options(sampling\_options);
 print\_options(spec\_options);
 print\_options(specific\_options);
 printf("\\"\\n\\n");

 printf(" case \\"$prev\\" in\\n");
 printf(" --model\|-m)\\n");
 printf(" COMPREPLY=( $(compgen -f -X '!\*.gguf' -- \\"$cur\\") $(compgen -d -- \\"$cur\\") )\\n");
 printf(" return 0\\n");
 printf(" ;;\\n");
 printf(" --grammar-file)\\n");
 printf(" COMPREPLY=( $(compgen -f -X '!\*.gbnf' -- \\"$cur\\") $(compgen -d -- \\"$cur\\") )\\n");
 printf(" return 0\\n");
 printf(" ;;\\n");
 printf(" --chat-template-file)\\n");
 printf(" COMPREPLY=( $(compgen -f -X '!\*.jinja' -- \\"$cur\\") $(compgen -d -- \\"$cur\\") )\\n");
 printf(" return 0\\n");
 printf(" ;;\\n");
 printf(" \*)\\n");
 printf(" COMPREPLY=( $(compgen -W \\"${opts}\\" -- \\"$cur\\") )\\n");
 printf(" return 0\\n");
 printf(" ;;\\n");
 printf(" esac\\n");
 printf("}\\n\\n");

 std::set executables = {
 "llama-batched",
 "llama-batched-bench",
 "llama-bench",
 "llama-cli",
 "llama-completion",
 "llama-convert-llama2c-to-ggml",
 "llama-cvector-generator",
 "llama-debug",
 "llama-diffusion-cli",
 "llama-embedding",
 "llama-eval-callback",
 "llama-export-lora",
 "llama-finetune",
 "llama-fit-params",
 "llama-gemma3-cli",
 "llama-gen-docs",
 "llama-gguf",
 "llama-gguf-hash",
 "llama-gguf-split",
 "llama-idle",
 "llama-imatrix",
 "llama-llava-cli",
 "llama-lookahead",
 "llama-lookup",
 "llama-lookup-create",
 "llama-lookup-merge",
 "llama-lookup-stats",
 "llama-minicpmv-cli",
 "llama-mtmd-cli",
 "llama-parallel",
 "llama-passkey",
 "llama-perplexity",
 "llama-q8dot",
 "llama-quantize",
 "llama-qwen2vl-cli",
 "llama-retrieval",
 "llama-save-load-state",
 "llama-server",
 "llama-simple",
 "llama-simple-chat",
 "llama-speculative",
 "llama-speculative-simple",
 "llama-tokenize",
 "llama-tts",
 "llama-vdot"
 };

 for (const auto& exe : executables) {
 printf("complete -F \_llama\_completions %s\\n", exe.c\_str());
 }
}

static std::vector parse\_device\_list(const std::string & value) {
 std::vector devices;
 auto dev\_names = string\_split(value, ',');
 if (dev\_names.empty()) {
 throw std::invalid\_argument("no devices specified");
 }
 if (dev\_names.size() == 1 && dev\_names\[0\] == "none") {
 devices.push\_back(nullptr);
 } else {
 ggml\_backend\_load\_all();
 for (const auto & device : dev\_names) {
 auto \* dev = ggml\_backend\_dev\_by\_name(device.c\_str());
 if (!dev \|\| ggml\_backend\_dev\_type(dev) == GGML\_BACKEND\_DEVICE\_TYPE\_CPU) {
 throw std::invalid\_argument(string\_format("invalid device: %s", device.c\_str()));
 }
 devices.push\_back(dev);
 }
 devices.push\_back(nullptr);
 }
 return devices;
}

static void add\_rpc\_devices(const std::string & servers) {
 auto rpc\_servers = string\_split(servers, ',');
 if (rpc\_servers.empty()) {
 throw std::invalid\_argument("no RPC servers specified");
 }
 ggml\_backend\_load\_all();
 ggml\_backend\_reg\_t rpc\_reg = ggml\_backend\_reg\_by\_name("RPC");
 if (!rpc\_reg) {
 throw std::invalid\_argument("failed to find RPC backend");
 }
 typedef ggml\_backend\_reg\_t (\*ggml\_backend\_rpc\_add\_server\_t)(const char \* endpoint);
 ggml\_backend\_rpc\_add\_server\_t ggml\_backend\_rpc\_add\_server\_fn = (ggml\_backend\_rpc\_add\_server\_t) ggml\_backend\_reg\_get\_proc\_address(rpc\_reg, "ggml\_backend\_rpc\_add\_server");
 if (!ggml\_backend\_rpc\_add\_server\_fn) {
 throw std::invalid\_argument("failed to find RPC add server function");
 }
 for (const auto & server : rpc\_servers) {
 auto reg = ggml\_backend\_rpc\_add\_server\_fn(server.c\_str());
 ggml\_backend\_register(reg);
 }
}

bool common\_params\_to\_map(int argc, char \*\* argv, llama\_example ex, std::map & out\_map) {
 common\_params dummy\_params;
 common\_params\_context ctx\_arg = common\_params\_parser\_init(dummy\_params, ex, nullptr);

 std::unordered\_map arg\_to\_options;
 for (auto & opt : ctx\_arg.options) {
 for (const auto & arg : opt.args) {
 arg\_to\_options\[arg\] = &opt;
 }
 for (const auto & arg : opt.args\_neg) {
 arg\_to\_options\[arg\] = &opt;
 }
 }

 // TODO @ngxson : find a way to deduplicate this code

 // handle command line arguments
 auto check\_arg = \[&\](int i) {
 if (i+1 >= argc) {
 throw std::invalid\_argument("expected value for argument");
 }
 };

 std::set seen\_args;

 for (int i = 1; i < argc; i++) {
 const std::string arg\_prefix = "--";

 std::string arg = argv\[i\];
 if (arg.compare(0, arg\_prefix.size(), arg\_prefix) == 0) {
 std::replace(arg.begin(), arg.end(), '\_', '-');
 }
 if (arg\_to\_options.find(arg) == arg\_to\_options.end()) {
 throw std::invalid\_argument(string\_format("error: invalid argument: %s", arg.c\_str()));
 }
 if (!seen\_args.insert(arg).second) {
 const bool skip = (arg == "--spec-type");

 if (!skip) {
 LOG\_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\\n", arg.c\_str());
 }
 }
 auto opt = \*arg\_to\_options\[arg\];
 std::string val;
 if (opt.value\_hint == nullptr && opt.value\_hint\_2 == nullptr) {
 // bool arg (need to reverse the meaning for negative args)
 bool is\_neg = std::find(opt.args\_neg.begin(), opt.args\_neg.end(), arg) != opt.args\_neg.end();
 val = is\_neg ? "0" : "1";
 }
 if (opt.value\_hint != nullptr) {
 // arg with single value
 check\_arg(i);
 val = argv\[++i\];
 }
 if (opt.value\_hint\_2 != nullptr) {
 // TODO: support arg with 2 values
 throw std::invalid\_argument("error: argument with 2 values is not yet supported\\n");
 }
 out\_map\[opt\] = val;
 }

 return true;
}

bool common\_params\_parse(int argc, char \*\* argv, common\_params & params, llama\_example ex, void(\*print\_usage)(int, char \*\*)) {
 auto ctx\_arg = common\_params\_parser\_init(params, ex, print\_usage);
 const common\_params params\_org = ctx\_arg.params; // the example can modify the default params

 try {
 if (!common\_params\_parse\_ex(argc, argv, ctx\_arg)) {
 ctx\_arg.params = params\_org;
 return false;
 }
 if (ctx\_arg.params.usage) {
 common\_params\_print\_usage(ctx\_arg);
 if (ctx\_arg.print\_usage) {
 ctx\_arg.print\_usage(argc, argv);
 }
 exit(0);
 }
 if (ctx\_arg.params.completion) {
 common\_params\_print\_completion(ctx\_arg);
 exit(0);
 }
 params.lr.init();
 } catch (const std::invalid\_argument & ex) {
 fprintf(stderr, "%s\\n", ex.what());
 ctx\_arg.params = params\_org;
 return false;
 } catch (std::exception & ex) {
 fprintf(stderr, "%s\\n", ex.what());
 exit(1); // for other exceptions, we exit with status code 1
 }

 return true;
}

static std::string list\_builtin\_chat\_templates() {
 std::vector supported\_tmpl;
 int32\_t res = llama\_chat\_builtin\_templates(nullptr, 0);
 supported\_tmpl.resize(res);
 res = llama\_chat\_builtin\_templates(supported\_tmpl.data(), supported\_tmpl.size());
 std::ostringstream msg;
 for (auto & tmpl : supported\_tmpl) {
 msg << tmpl << (&tmpl == &supported\_tmpl.back() ? "" : ", ");
 }
 return msg.str();
}

bool common\_arg\_utils::is\_truthy(const std::string & value) {
 return value == "on" \|\| value == "enabled" \|\| value == "true" \|\| value == "1";
}

bool common\_arg\_utils::is\_falsey(const std::string & value) {
 return value == "off" \|\| value == "disabled" \|\| value == "false" \|\| value == "0";
}

bool common\_arg\_utils::is\_autoy(const std::string & value) {
 return value == "auto" \|\| value == "-1";
}

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
// input: value1,"value, with, commas","value with ""escaped"" quotes",value4
// output: \[value1\] \[value, with, commas\] \[value with "escaped" quotes\] \[value4\]
static std::vector parse\_csv\_row(const std::string& input) {
 std::vector fields;
 std::string field;
 bool in\_quotes = false;

 for (size\_t i = 0; i < input.length(); ++i) {
 char ch = input\[i\];

 if (ch == '"') {
 if (!in\_quotes) {
 // start of quoted field (only valid if at beginning of field)
 if (!field.empty()) {
 // quote appeared in middle of unquoted field, treat as literal
 field += '"';
 } else {
 in\_quotes = true; // start
 }
 } else {
 if (i + 1 < input.length() && input\[i + 1\] == '"') {
 // escaped quote: ""
 field += '"';
 ++i; // skip the next quote
 } else {
 in\_quotes = false; // end
 }
 }
 } else if (ch == ',') {
 if (in\_quotes) {
 field += ',';
 } else {
 fields.push\_back(std::move(field));
 field.clear();
 }
 } else {
 field += ch;
 }
 }

 // Add the last field
 fields.push\_back(std::move(field));

 return fields;
}

common\_params\_context common\_params\_parser\_init(common\_params & params, llama\_example ex, void(\*print\_usage)(int, char \*\*)) {
 // per-example default params
 // we define here to make sure it's included in llama-gen-docs
 if (ex == LLAMA\_EXAMPLE\_COMPLETION) {
 params.use\_jinja = false; // disable jinja by default

 } else if (ex == LLAMA\_EXAMPLE\_MTMD) {
 params.use\_jinja = false; // disable jinja by default
 params.sampling.temp = 0.2; // lower temp by default for better quality

 } else if (ex == LLAMA\_EXAMPLE\_SERVER) {
 params.n\_parallel = -1; // auto by default
 }

 params.use\_color = tty\_can\_use\_colors();

 common\_params\_context ctx\_arg(params);
 ctx\_arg.print\_usage = print\_usage;
 ctx\_arg.ex = ex;

 std::string sampler\_type\_chars;
 std::string sampler\_type\_names;
 for (const auto & sampler : params.sampling.samplers) {
 sampler\_type\_chars += common\_sampler\_type\_to\_chr(sampler);
 sampler\_type\_names += common\_sampler\_type\_to\_str(sampler) + ";";
 }
 if (!sampler\_type\_names.empty()) {
 sampler\_type\_names.pop\_back(); // remove last semicolon
 }

 /\\*\\*
 \\* filter options by example
 \\* rules:
 \\* \- all examples inherit options from LLAMA\_EXAMPLE\_COMMON
 \\* \- if LLAMA\_EXAMPLE\_\* is set (other than COMMON), we only show the option in the corresponding example
 \\* \- if both {LLAMA\_EXAMPLE\_COMMON, LLAMA\_EXAMPLE\_\*,} are set, we will prioritize the LLAMA\_EXAMPLE\_\* matching current example
 \*/
 auto add\_opt = \[&\](common\_arg arg) {
 if ((arg.in\_example(ex) \|\| arg.in\_example(LLAMA\_EXAMPLE\_COMMON)) && !arg.is\_exclude(ex)) {
 ctx\_arg.options.push\_back(std::move(arg));
 }
 };

 add\_opt(common\_arg(
 {"-h", "--help", "--usage"},
 "print usage and exit",
 \[\](common\_params & params) {
 params.usage = true;
 }
 ));
 add\_opt(common\_arg(
 {"--version"},
 "show version and build info",
 \[\](common\_params &) {
 fprintf(stderr, "version: %d (%s)\\n", llama\_build\_number(), llama\_commit());
 fprintf(stderr, "built with %s for %s\\n", llama\_compiler(), llama\_build\_target());
 exit(0);
 }
 ));
 add\_opt(common\_arg(
 {"--license"},
 "show source code license and dependencies",
 \[\](common\_params &) {
 for (int i = 0; LICENSES\[i\]; ++i) {
 printf("%s\\n", LICENSES\[i\]);
 }
 exit(0);
 }
 ));
 add\_opt(common\_arg(
 {"-cl", "--cache-list"},
 "show list of models in cache",
 \[\](common\_params &) {
 auto models = common\_list\_cached\_models();
 printf("number of models in cache: %zu\\n", models.size());
 for (size\_t i = 0; i < models.size(); i++) {
 printf("%4zu. %s\\n", i + 1, models\[i\].to\_string().c\_str());
 }
 exit(0);
 }
 ));
 add\_opt(common\_arg(
 {"--completion-bash"},
 "print source-able bash completion script for llama.cpp",
 \[\](common\_params & params) {
 params.completion = true;
 }
 ));
 add\_opt(common\_arg(
 {"--verbose-prompt"},
 string\_format("print a verbose prompt before generation (default: %s)", params.verbose\_prompt ? "true" : "false"),
 \[\](common\_params & params) {
 params.verbose\_prompt = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_EMBEDDING, LLAMA\_EXAMPLE\_RETRIEVAL}));
 add\_opt(common\_arg(
 {"--display-prompt"},
 {"--no-display-prompt"},
 string\_format("whether to print prompt at generation (default: %s)", params.display\_prompt ? "true" : "false"),
 \[\](common\_params & params, bool value) {
 params.display\_prompt = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"-co", "--color"}, "\[on\|off\|auto\]",
 "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\\n"
 "'auto' enables colors when output is to a terminal",
 \[\](common\_params & params, const std::string & value) {
 if (is\_truthy(value)) {
 params.use\_color = true;
 } else if (is\_falsey(value)) {
 params.use\_color = false;
 } else if (is\_autoy(value)) {
 params.use\_color = tty\_can\_use\_colors();
 } else {
 throw std::invalid\_argument(
 string\_format("error: unknown value for --color: '%s'\\n", value.c\_str()));
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_LOOKUP}));
 add\_opt(common\_arg(
 {"-t", "--threads"}, "N",
 string\_format("number of CPU threads to use during generation (default: %d)", params.cpuparams.n\_threads),
 \[\](common\_params & params, int value) {
 params.cpuparams.n\_threads = value;
 if (params.cpuparams.n\_threads <= 0) {
 params.cpuparams.n\_threads = std::thread::hardware\_concurrency();
 }
 }
 ).set\_env("LLAMA\_ARG\_THREADS"));
 add\_opt(common\_arg(
 {"-tb", "--threads-batch"}, "N",
 "number of threads to use during batch and prompt processing (default: same as --threads)",
 \[\](common\_params & params, int value) {
 params.cpuparams\_batch.n\_threads = value;
 if (params.cpuparams\_batch.n\_threads <= 0) {
 params.cpuparams\_batch.n\_threads = std::thread::hardware\_concurrency();
 }
 }
 ));
 add\_opt(common\_arg(
 {"-C", "--cpu-mask"}, "M",
 "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \\"\\")",
 \[\](common\_params & params, const std::string & mask) {
 params.cpuparams.mask\_valid = true;
 if (!parse\_cpu\_mask(mask, params.cpuparams.cpumask)) {
 throw std::invalid\_argument("invalid cpumask");
 }
 }
 ));
 add\_opt(common\_arg(
 {"-Cr", "--cpu-range"}, "lo-hi",
 "range of CPUs for affinity. Complements --cpu-mask",
 \[\](common\_params & params, const std::string & range) {
 params.cpuparams.mask\_valid = true;
 if (!parse\_cpu\_range(range, params.cpuparams.cpumask)) {
 throw std::invalid\_argument("invalid range");
 }
 }
 ));
 add\_opt(common\_arg(
 {"--cpu-strict"}, "<0\|1>",
 string\_format("use strict CPU placement (default: %u)\\n", (unsigned) params.cpuparams.strict\_cpu),
 \[\](common\_params & params, const std::string & value) {
 params.cpuparams.strict\_cpu = std::stoul(value);
 }
 ));
 add\_opt(common\_arg(
 {"--prio"}, "N",
 string\_format("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\\n", params.cpuparams.priority),
 \[\](common\_params & params, int prio) {
 if (prio < GGML\_SCHED\_PRIO\_LOW \|\| prio > GGML\_SCHED\_PRIO\_REALTIME) {
 throw std::invalid\_argument("invalid value");
 }
 params.cpuparams.priority = (enum ggml\_sched\_priority) prio;
 }
 ));
 add\_opt(common\_arg(
 {"--poll"}, "<0...100>",
 string\_format("use polling level to wait for work (0 - no polling, default: %u)\\n", (unsigned) params.cpuparams.poll),
 \[\](common\_params & params, const std::string & value) {
 params.cpuparams.poll = std::stoul(value);
 }
 ));
 add\_opt(common\_arg(
 {"-Cb", "--cpu-mask-batch"}, "M",
 "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
 \[\](common\_params & params, const std::string & mask) {
 params.cpuparams\_batch.mask\_valid = true;
 if (!parse\_cpu\_mask(mask, params.cpuparams\_batch.cpumask)) {
 throw std::invalid\_argument("invalid cpumask");
 }
 }
 ));
 add\_opt(common\_arg(
 {"-Crb", "--cpu-range-batch"}, "lo-hi",
 "ranges of CPUs for affinity. Complements --cpu-mask-batch",
 \[\](common\_params & params, const std::string & range) {
 params.cpuparams\_batch.mask\_valid = true;
 if (!parse\_cpu\_range(range, params.cpuparams\_batch.cpumask)) {
 throw std::invalid\_argument("invalid range");
 }
 }
 ));
 add\_opt(common\_arg(
 {"--cpu-strict-batch"}, "<0\|1>",
 "use strict CPU placement (default: same as --cpu-strict)",
 \[\](common\_params & params, int value) {
 params.cpuparams\_batch.strict\_cpu = value;
 }
 ));
 add\_opt(common\_arg(
 {"--prio-batch"}, "N",
 string\_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\\n", params.cpuparams\_batch.priority),
 \[\](common\_params & params, int prio) {
 if (prio < 0 \|\| prio > 3) {
 throw std::invalid\_argument("invalid value");
 }
 params.cpuparams\_batch.priority = (enum ggml\_sched\_priority) prio;
 }
 ));
 add\_opt(common\_arg(
 {"--poll-batch"}, "<0\|1>",
 "use polling to wait for work (default: same as --poll)",
 \[\](common\_params & params, int value) {
 params.cpuparams\_batch.poll = value;
 }
 ));
 add\_opt(common\_arg(
 {"-lcs", "--lookup-cache-static"}, "FNAME",
 "path to static lookup cache to use for lookup decoding (not updated by generation)",
 \[\](common\_params & params, const std::string & value) {
 params.speculative.ngram\_cache.lookup\_cache\_static = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_LOOKUP, LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
 "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
 \[\](common\_params & params, const std::string & value) {
 params.speculative.ngram\_cache.lookup\_cache\_dynamic = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_LOOKUP, LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-c", "--ctx-size"}, "N",
 string\_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n\_ctx),
 \[\](common\_params & params, int value) {
 params.n\_ctx = value;
 if (value == 0) {
 // disable context reduction in llama\_params\_fit if the user explicitly requests the full context size:
 params.fit\_params\_min\_ctx = UINT32\_MAX;
 }
 }
 ).set\_env("LLAMA\_ARG\_CTX\_SIZE"));
 add\_opt(common\_arg(
 {"-n", "--predict", "--n-predict"}, "N",
 string\_format(
 ex == LLAMA\_EXAMPLE\_COMPLETION
 ? "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)"
 : "number of tokens to predict (default: %d, -1 = infinity)",
 params.n\_predict),
 \[\](common\_params & params, int value) {
 params.n\_predict = value;
 }
 ).set\_env("LLAMA\_ARG\_N\_PREDICT"));
 add\_opt(common\_arg(
 {"-b", "--batch-size"}, "N",
 string\_format("logical maximum batch size (default: %d)", params.n\_batch),
 \[\](common\_params & params, int value) {
 params.n\_batch = value;
 }
 ).set\_env("LLAMA\_ARG\_BATCH"));
 add\_opt(common\_arg(
 {"-ub", "--ubatch-size"}, "N",
 string\_format("physical maximum batch size (default: %d)", params.n\_ubatch),
 \[\](common\_params & params, int value) {
 params.n\_ubatch = value;
 }
 ).set\_env("LLAMA\_ARG\_UBATCH"));
 add\_opt(common\_arg(
 {"--keep"}, "N",
 string\_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n\_keep),
 \[\](common\_params & params, int value) {
 params.n\_keep = value;
 }
 ));
 add\_opt(common\_arg(
 {"--swa-full"},
 string\_format("use full-size SWA cache (default: %s)\\n"
 "\[(more info)\](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa\_full ? "true" : "false"),
 \[\](common\_params & params) {
 params.swa\_full = true;
 }
 ).set\_env("LLAMA\_ARG\_SWA\_FULL"));
 add\_opt(common\_arg(
 {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}, "N",
 string\_format("max number of context checkpoints to create per slot (default: %d)"
 "\[(more info)\](https://github.com/ggml-org/llama.cpp/pull/15293)", params.n\_ctx\_checkpoints),
 \[\](common\_params & params, int value) {
 params.n\_ctx\_checkpoints = value;
 }
 ).set\_env("LLAMA\_ARG\_CTX\_CHECKPOINTS").set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"-cpent", "--checkpoint-every-n-tokens"}, "N",
 string\_format("create a checkpoint every n tokens during prefill (processing), -1 to disable (default: %d)", params.checkpoint\_every\_nt),
 \[\](common\_params & params, int value) {
 params.checkpoint\_every\_nt = value;
 }
 ).set\_env("LLAMA\_ARG\_CHECKPOINT\_EVERY\_NT").set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"-cram", "--cache-ram"}, "N",
 string\_format("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)"
 "\[(more info)\](https://github.com/ggml-org/llama.cpp/pull/16391)", params.cache\_ram\_mib),
 \[\](common\_params & params, int value) {
 params.cache\_ram\_mib = value;
 }
 ).set\_env("LLAMA\_ARG\_CACHE\_RAM").set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"-kvu", "--kv-unified"},
 {"-no-kvu", "--no-kv-unified"},
 "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)",
 \[\](common\_params & params, bool value) {
 params.kv\_unified = value;
 }
 ).set\_env("LLAMA\_ARG\_KV\_UNIFIED").set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_PERPLEXITY, LLAMA\_EXAMPLE\_BATCHED, LLAMA\_EXAMPLE\_BENCH, LLAMA\_EXAMPLE\_PARALLEL}));
 add\_opt(common\_arg(
 {"--cache-idle-slots"},
 {"--no-cache-idle-slots"},
 "save and clear idle slots on new task (default: enabled, requires unified KV and cache-ram)",
 \[\](common\_params & params, bool value) {
 params.cache\_idle\_slots = value;
 }
 ).set\_env("LLAMA\_ARG\_CACHE\_IDLE\_SLOTS").set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--context-shift"},
 {"--no-context-shift"},
 string\_format("whether to use context shift on infinite text generation (default: %s)", params.ctx\_shift ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.ctx\_shift = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_IMATRIX, LLAMA\_EXAMPLE\_PERPLEXITY}).set\_env("LLAMA\_ARG\_CONTEXT\_SHIFT"));
 add\_opt(common\_arg(
 {"--chunks"}, "N",
 string\_format("max number of chunks to process (default: %d, -1 = all)", params.n\_chunks),
 \[\](common\_params & params, int value) {
 params.n\_chunks = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX, LLAMA\_EXAMPLE\_PERPLEXITY, LLAMA\_EXAMPLE\_RETRIEVAL}));
 add\_opt(common\_arg({ "-fa", "--flash-attn" }, "\[on\|off\|auto\]",
 string\_format("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
 llama\_flash\_attn\_type\_name(params.flash\_attn\_type)),
 \[\](common\_params & params, const std::string & value) {
 if (is\_truthy(value)) {
 params.flash\_attn\_type = LLAMA\_FLASH\_ATTN\_TYPE\_ENABLED;
 } else if (is\_falsey(value)) {
 params.flash\_attn\_type = LLAMA\_FLASH\_ATTN\_TYPE\_DISABLED;
 } else if (is\_autoy(value)) {
 params.flash\_attn\_type = LLAMA\_FLASH\_ATTN\_TYPE\_AUTO;
 } else {
 throw std::runtime\_error(
 string\_format("error: unknown value for --flash-attn: '%s'\\n", value.c\_str()));
 }
 }).set\_env("LLAMA\_ARG\_FLASH\_ATTN"));
 add\_opt(common\_arg(
 {"-p", "--prompt"}, "PROMPT",
 "prompt to start generation with; for system message, use -sys",
 \[\](common\_params & params, const std::string & value) {
 params.prompt = value;
 }
 ).set\_excludes({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-sys", "--system-prompt"}, "PROMPT",
 "system prompt to use with model (if applicable, depending on chat template)",
 \[\](common\_params & params, const std::string & value) {
 params.system\_prompt = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_DIFFUSION, LLAMA\_EXAMPLE\_MTMD}));
 add\_opt(common\_arg(
 {"--perf"},
 {"--no-perf"},
 string\_format("whether to enable internal libllama performance timings (default: %s)", params.no\_perf ? "true" : "false"),
 \[\](common\_params & params, bool value) {
 params.no\_perf = !value;
 params.sampling.no\_perf = !value;
 }
 ).set\_env("LLAMA\_ARG\_PERF"));
 add\_opt(common\_arg(
 {"--show-timings"},
 {"--no-show-timings"},
 string\_format("whether to show timing information after each response (default: %s)", params.show\_timings ? "true" : "false"),
 \[\](common\_params & params, bool value) {
 params.show\_timings = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SHOW\_TIMINGS"));
 add\_opt(common\_arg(
 {"-f", "--file"}, "FNAME",
 "a file containing the prompt (default: none)",
 \[\](common\_params & params, const std::string & value) {
 params.prompt = read\_file(value);
 // store the external file name in params
 params.prompt\_file = value;
 if (!params.prompt.empty() && params.prompt.back() == '\\n') {
 params.prompt.pop\_back();
 }
 }
 ).set\_excludes({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-sysf", "--system-prompt-file"}, "FNAME",
 "a file containing the system prompt (default: none)",
 \[\](common\_params & params, const std::string & value) {
 params.system\_prompt = read\_file(value);
 if (!params.system\_prompt.empty() && params.system\_prompt.back() == '\\n') {
 params.system\_prompt.pop\_back();
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_DIFFUSION}));
 add\_opt(common\_arg(
 {"--in-file"}, "FNAME",
 "an input file (use comma-separated values to specify multiple files)",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 std::ifstream file(item);
 if (!file) {
 throw std::runtime\_error(string\_format("error: failed to open file '%s'\\n", item.c\_str()));
 }
 params.in\_files.push\_back(item);
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"-bf", "--binary-file"}, "FNAME",
 "binary file containing the prompt (default: none)",
 \[\](common\_params & params, const std::string & value) {
 std::ifstream file(value, std::ios::binary);
 if (!file) {
 throw std::runtime\_error(string\_format("error: failed to open file '%s'\\n", value.c\_str()));
 }
 // store the external file name in params
 params.prompt\_file = value;
 std::ostringstream ss;
 ss << file.rdbuf();
 params.prompt = ss.str();
 fprintf(stderr, "Read %zu bytes from binary file %s\\n", params.prompt.size(), value.c\_str());
 }
 ).set\_excludes({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-e", "--escape"},
 {"--no-escape"},
 string\_format("whether to process escapes sequences (\\\n, \\\r, \\\t, \\\', \\\\\", \\\\\\) (default: %s)", params.escape ? "true" : "false"),
 \[\](common\_params & params, bool value) {
 params.escape = value;
 }
 ));
 add\_opt(common\_arg(
 {"-ptc", "--print-token-count"}, "N",
 string\_format("print token count every N tokens (default: %d)", params.n\_print),
 \[\](common\_params & params, int value) {
 params.n\_print = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"--prompt-cache"}, "FNAME",
 "file to cache prompt state for faster startup (default: none)",
 \[\](common\_params & params, const std::string & value) {
 params.path\_prompt\_cache = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"--prompt-cache-all"},
 "if specified, saves user input and generations to cache as well\\n",
 \[\](common\_params & params) {
 params.prompt\_cache\_all = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"--prompt-cache-ro"},
 "if specified, uses the prompt cache but does not update it",
 \[\](common\_params & params) {
 params.prompt\_cache\_ro = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"-r", "--reverse-prompt"}, "PROMPT",
 "halt generation at PROMPT, return control in interactive mode\\n",
 \[\](common\_params & params, const std::string & value) {
 params.antiprompt.emplace\_back(value);
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-sp", "--special"},
 string\_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
 \[\](common\_params & params) {
 params.special = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"-cnv", "--conversation"},
 {"-no-cnv", "--no-conversation"},
 "whether to run in conversation mode:\\n"
 "\- does not print special tokens and suffix/prefix\\n"
 "\- interactive mode is also enabled\\n"
 "(default: auto enabled if chat template is available)",
 \[\](common\_params & params, bool value) {
 params.conversation\_mode = value ? COMMON\_CONVERSATION\_MODE\_ENABLED : COMMON\_CONVERSATION\_MODE\_DISABLED;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"-st", "--single-turn"},
 "run conversation for a single turn only, then exit when done\\n"
 "will not be interactive if first turn is predefined with --prompt\\n"
 "(default: false)",
 \[\](common\_params & params) {
 params.single\_turn = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"-i", "--interactive"},
 string\_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
 \[\](common\_params & params) {
 params.interactive = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"-if", "--interactive-first"},
 string\_format("run in interactive mode and wait for input right away (default: %s)", params.interactive\_first ? "true" : "false"),
 \[\](common\_params & params) {
 params.interactive\_first = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"-mli", "--multiline-input"},
 "allows you to write or paste multiple lines without ending each in '\\\'",
 \[\](common\_params & params) {
 params.multiline\_input = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--in-prefix-bos"},
 "prefix BOS to user inputs, preceding the \`--in-prefix\` string",
 \[\](common\_params & params) {
 params.input\_prefix\_bos = true;
 params.enable\_chat\_template = false;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"--in-prefix"}, "STRING",
 "string to prefix user inputs with (default: empty)",
 \[\](common\_params & params, const std::string & value) {
 params.input\_prefix = value;
 params.enable\_chat\_template = false;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"--in-suffix"}, "STRING",
 "string to suffix after user inputs with (default: empty)",
 \[\](common\_params & params, const std::string & value) {
 params.input\_suffix = value;
 params.enable\_chat\_template = false;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"--warmup"},
 {"--no-warmup"},
 string\_format("whether to perform warmup with an empty run (default: %s)", params.warmup ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.warmup = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_MTMD, LLAMA\_EXAMPLE\_EMBEDDING, LLAMA\_EXAMPLE\_RETRIEVAL, LLAMA\_EXAMPLE\_PERPLEXITY, LLAMA\_EXAMPLE\_DEBUG}));
 add\_opt(common\_arg(
 {"--spm-infill"},
 string\_format(
 "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
 params.spm\_infill ? "enabled" : "disabled"
 ),
 \[\](common\_params & params) {
 params.spm\_infill = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--samplers"}, "SAMPLERS",
 string\_format("samplers that will be used for generation in the order, separated by \\';\\'\\n(default: %s)", sampler\_type\_names.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 const auto sampler\_names = string\_split(value, ';');
 params.sampling.samplers = common\_sampler\_types\_from\_names(sampler\_names, true);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_SAMPLERS;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"-s", "--seed"}, "SEED",
 string\_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA\_DEFAULT\_SEED),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.seed = std::stoul(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--sampler-seq", "--sampling-seq"}, "SEQUENCE",
 string\_format("simplified sequence for samplers that will be used (default: %s)", sampler\_type\_chars.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.samplers = common\_sampler\_types\_from\_chars(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--ignore-eos"},
 "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
 \[\](common\_params & params) {
 params.sampling.ignore\_eos = true;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--temp", "--temperature"}, "N",
 string\_format("temperature (default: %.2f)", (double)params.sampling.temp),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.temp = std::stof(value);
 params.sampling.temp = std::max(params.sampling.temp, 0.0f);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_TEMP;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--top-k"}, "N",
 string\_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top\_k),
 \[\](common\_params & params, int value) {
 params.sampling.top\_k = value;
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_TOP\_K;
 }
 ).set\_sampling().set\_env("LLAMA\_ARG\_TOP\_K"));
 add\_opt(common\_arg(
 {"--top-p"}, "N",
 string\_format("top-p sampling (default: %.2f, 1.0 = disabled)", (double)params.sampling.top\_p),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.top\_p = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_TOP\_P;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--min-p"}, "N",
 string\_format("min-p sampling (default: %.2f, 0.0 = disabled)", (double)params.sampling.min\_p),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.min\_p = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_MIN\_P;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--top-nsigma", "--top-n-sigma"}, "N",
 string\_format("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", params.sampling.top\_n\_sigma),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.top\_n\_sigma = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--xtc-probability"}, "N",
 string\_format("xtc probability (default: %.2f, 0.0 = disabled)", (double)params.sampling.xtc\_probability),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.xtc\_probability = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_XTC\_PROBABILITY;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--xtc-threshold"}, "N",
 string\_format("xtc threshold (default: %.2f, 1.0 = disabled)", (double)params.sampling.xtc\_threshold),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.xtc\_threshold = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_XTC\_THRESHOLD;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--typical", "--typical-p"}, "N",
 string\_format("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)", (double)params.sampling.typ\_p),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.typ\_p = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--repeat-last-n"}, "N",
 string\_format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx\_size)", params.sampling.penalty\_last\_n),
 \[\](common\_params & params, int value) {
 if (value < -1) {
 throw std::runtime\_error(string\_format("error: invalid repeat-last-n = %d\\n", value));
 }
 params.sampling.penalty\_last\_n = value;
 params.sampling.n\_prev = std::max(params.sampling.n\_prev, params.sampling.penalty\_last\_n);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_PENALTY\_LAST\_N;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--repeat-penalty"}, "N",
 string\_format("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)", (double)params.sampling.penalty\_repeat),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.penalty\_repeat = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_PENALTY\_REPEAT;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--presence-penalty"}, "N",
 string\_format("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty\_present),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.penalty\_present = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--frequency-penalty"}, "N",
 string\_format("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty\_freq),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.penalty\_freq = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dry-multiplier"}, "N",
 string\_format("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)", (double)params.sampling.dry\_multiplier),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.dry\_multiplier = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dry-base"}, "N",
 string\_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry\_base),
 \[\](common\_params & params, const std::string & value) {
 float potential\_base = std::stof(value);
 if (potential\_base >= 1.0f)
 {
 params.sampling.dry\_base = potential\_base;
 }
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dry-allowed-length"}, "N",
 string\_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry\_allowed\_length),
 \[\](common\_params & params, int value) {
 params.sampling.dry\_allowed\_length = value;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dry-penalty-last-n"}, "N",
 string\_format("set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", params.sampling.dry\_penalty\_last\_n),
 \[\](common\_params & params, int value) {
 if (value < -1) {
 throw std::runtime\_error(string\_format("error: invalid dry-penalty-last-n = %d\\n", value));
 }
 params.sampling.dry\_penalty\_last\_n = value;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dry-sequence-breaker"}, "STRING",
 string\_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \\"none\\" to not use any sequence breakers\\n",
 params.sampling.dry\_sequence\_breakers.empty() ? "none" :
 std::accumulate(std::next(params.sampling.dry\_sequence\_breakers.begin()),
 params.sampling.dry\_sequence\_breakers.end(),
 std::string("'") + (params.sampling.dry\_sequence\_breakers\[0\] == "\\n" ? "\\\n" : params.sampling.dry\_sequence\_breakers\[0\]) + "'",
 \[\](const std::string& a, const std::string& b) {
 std::string formatted\_b = (b == "\\n") ? "\\\n" : b;
 return a + ", '" + formatted\_b + "'";
 }).c\_str()),
 \[\](common\_params & params, const std::string & value) {
 static bool defaults\_cleared = false;

 if (!defaults\_cleared) {
 params.sampling.dry\_sequence\_breakers.clear();
 defaults\_cleared = true;
 }

 if (value == "none") {
 params.sampling.dry\_sequence\_breakers.clear();
 } else {
 params.sampling.dry\_sequence\_breakers.emplace\_back(value);
 }
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--adaptive-target"}, "N",
 string\_format("adaptive-p: select tokens near this probability (valid range 0.0 "
 "to 1.0; negative = disabled) (default: %.2f)\\n"
 "\[(more info)\](https://github.com/ggml-org/llama.cpp/pull/17927)",
 (double)params.sampling.adaptive\_target),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.adaptive\_target = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--adaptive-decay"}, "N",
 string\_format("adaptive-p: decay rate for target adaptation over time. lower values "
 "are more reactive, higher values are more stable.\\n"
 "(valid range 0.0 to 0.99) (default: %.2f)",
 (double)params.sampling.adaptive\_decay),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.adaptive\_decay = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dynatemp-range"}, "N",
 string\_format("dynamic temperature range (default: %.2f, 0.0 = disabled)", (double)params.sampling.dynatemp\_range),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.dynatemp\_range = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--dynatemp-exp"}, "N",
 string\_format("dynamic temperature exponent (default: %.2f)", (double)params.sampling.dynatemp\_exponent),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.dynatemp\_exponent = std::stof(value);
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--mirostat"}, "N",
 string\_format("use Mirostat sampling.\\nTop K, Nucleus and Locally Typical samplers are ignored if used.\\n"
 "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
 \[\](common\_params & params, int value) {
 params.sampling.mirostat = value;
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_MIROSTAT;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--mirostat-lr"}, "N",
 string\_format("Mirostat learning rate, parameter eta (default: %.2f)", (double)params.sampling.mirostat\_eta),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.mirostat\_eta = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_MIROSTAT\_ETA;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--mirostat-ent"}, "N",
 string\_format("Mirostat target entropy, parameter tau (default: %.2f)", (double)params.sampling.mirostat\_tau),
 \[\](common\_params & params, const std::string & value) {
 params.sampling.mirostat\_tau = std::stof(value);
 params.sampling.user\_sampling\_config \|= common\_params\_sampling\_config::COMMON\_PARAMS\_SAMPLING\_CONFIG\_MIROSTAT\_TAU;
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"-l", "--logit-bias"}, "TOKEN\_ID(+/-)BIAS",
 "modifies the likelihood of token appearing in the completion,\\n"
 "i.e. \`--logit-bias 15043+1\` to increase likelihood of token ' Hello',\\n"
 "or \`--logit-bias 15043-1\` to decrease likelihood of token ' Hello'",
 \[\](common\_params & params, const std::string & value) {
 std::stringstream ss(value);
 llama\_token key;
 char sign;
 std::string value\_str;
 try {
 if (ss >> key && ss >> sign && std::getline(ss, value\_str) && (sign == '+' \|\| sign == '-')) {
 const float bias = std::stof(value\_str) \* ((sign == '-') ? -1.0f : 1.0f);
 params.sampling.logit\_bias.push\_back({key, bias});
 } else {
 throw std::invalid\_argument("invalid input format");
 }
 } catch (const std::exception&) {
 throw std::invalid\_argument("invalid input format");
 }
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--grammar"}, "GRAMMAR",
 "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
 \[\](common\_params & params, const std::string & value) {
 params.sampling.grammar = {COMMON\_GRAMMAR\_TYPE\_USER, value};
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"--grammar-file"}, "FNAME",
 "file to read grammar from",
 \[\](common\_params & params, const std::string & value) {
 params.sampling.grammar = {COMMON\_GRAMMAR\_TYPE\_USER, read\_file(value)};
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"-j", "--json-schema"}, "SCHEMA",
 "JSON schema to constrain generations (https://json-schema.org/), e.g. \`{}\` for any JSON object\\nFor schemas w/ external $refs, use --grammar + example/json\_schema\_to\_grammar.py instead",
 \[\](common\_params & params, const std::string & value) {
 params.sampling.grammar = {COMMON\_GRAMMAR\_TYPE\_OUTPUT\_FORMAT, json\_schema\_to\_grammar(json::parse(value))};
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"-jf", "--json-schema-file"}, "FILE",
 "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. \`{}\` for any JSON object\\nFor schemas w/ external $refs, use --grammar + example/json\_schema\_to\_grammar.py instead",
 \[\](common\_params & params, const std::string & value) {
 std::ifstream file(value);
 if (!file) {
 throw std::runtime\_error(string\_format("error: failed to open file '%s'\\n", value.c\_str()));
 }
 std::string schema;
 std::copy(
 std::istreambuf\_iterator(file),
 std::istreambuf\_iterator(),
 std::back\_inserter(schema)
 );
 params.sampling.grammar = {COMMON\_GRAMMAR\_TYPE\_OUTPUT\_FORMAT, json\_schema\_to\_grammar(json::parse(schema))};
 }
 ).set\_sampling());
 add\_opt(common\_arg(
 {"-bs", "--backend-sampling"},
 "enable backend sampling (experimental) (default: disabled)",
 \[\](common\_params & params) {
 params.sampling.backend\_sampling = true;
 }
 ).set\_sampling().set\_env("LLAMA\_ARG\_BACKEND\_SAMPLING"));
 add\_opt(common\_arg(
 {"--pooling"}, "{none,mean,cls,last,rank}",
 "pooling type for embeddings, use model default if unspecified",
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "none") { params.pooling\_type = LLAMA\_POOLING\_TYPE\_NONE; }
 else if (value == "mean") { params.pooling\_type = LLAMA\_POOLING\_TYPE\_MEAN; }
 else if (value == "cls") { params.pooling\_type = LLAMA\_POOLING\_TYPE\_CLS; }
 else if (value == "last") { params.pooling\_type = LLAMA\_POOLING\_TYPE\_LAST; }
 else if (value == "rank") { params.pooling\_type = LLAMA\_POOLING\_TYPE\_RANK; }
 else { throw std::invalid\_argument("invalid value"); }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING, LLAMA\_EXAMPLE\_RETRIEVAL, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_DEBUG}).set\_env("LLAMA\_ARG\_POOLING"));
 add\_opt(common\_arg(
 {"--attention"}, "{causal,non-causal}",
 "attention type for embeddings, use model default if unspecified",
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "causal") { params.attention\_type = LLAMA\_ATTENTION\_TYPE\_CAUSAL; }
 else if (value == "non-causal") { params.attention\_type = LLAMA\_ATTENTION\_TYPE\_NON\_CAUSAL; }
 else { throw std::invalid\_argument("invalid value"); }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING}));
 add\_opt(common\_arg(
 {"--rope-scaling"}, "{none,linear,yarn}",
 "RoPE frequency scaling method, defaults to linear unless specified by the model",
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "none") { params.rope\_scaling\_type = LLAMA\_ROPE\_SCALING\_TYPE\_NONE; }
 else if (value == "linear") { params.rope\_scaling\_type = LLAMA\_ROPE\_SCALING\_TYPE\_LINEAR; }
 else if (value == "yarn") { params.rope\_scaling\_type = LLAMA\_ROPE\_SCALING\_TYPE\_YARN; }
 else { throw std::invalid\_argument("invalid value"); }
 }
 ).set\_env("LLAMA\_ARG\_ROPE\_SCALING\_TYPE"));
 add\_opt(common\_arg(
 {"--rope-scale"}, "N",
 "RoPE context scaling factor, expands context by a factor of N",
 \[\](common\_params & params, const std::string & value) {
 params.rope\_freq\_scale = 1.0f / std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_ROPE\_SCALE"));
 add\_opt(common\_arg(
 {"--rope-freq-base"}, "N",
 "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
 \[\](common\_params & params, const std::string & value) {
 params.rope\_freq\_base = std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_ROPE\_FREQ\_BASE"));
 add\_opt(common\_arg(
 {"--rope-freq-scale"}, "N",
 "RoPE frequency scaling factor, expands context by a factor of 1/N",
 \[\](common\_params & params, const std::string & value) {
 params.rope\_freq\_scale = std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_ROPE\_FREQ\_SCALE"));
 add\_opt(common\_arg(
 {"--yarn-orig-ctx"}, "N",
 string\_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn\_orig\_ctx),
 \[\](common\_params & params, int value) {
 params.yarn\_orig\_ctx = value;
 }
 ).set\_env("LLAMA\_ARG\_YARN\_ORIG\_CTX"));
 add\_opt(common\_arg(
 {"--yarn-ext-factor"}, "N",
 string\_format("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)", (double)params.yarn\_ext\_factor),
 \[\](common\_params & params, const std::string & value) {
 params.yarn\_ext\_factor = std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_YARN\_EXT\_FACTOR"));
 add\_opt(common\_arg(
 {"--yarn-attn-factor"}, "N",
 string\_format("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)params.yarn\_attn\_factor),
 \[\](common\_params & params, const std::string & value) {
 params.yarn\_attn\_factor = std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_YARN\_ATTN\_FACTOR"));
 add\_opt(common\_arg(
 {"--yarn-beta-slow"}, "N",
 string\_format("YaRN: high correction dim or alpha (default: %.2f)", (double)params.yarn\_beta\_slow),
 \[\](common\_params & params, const std::string & value) {
 params.yarn\_beta\_slow = std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_YARN\_BETA\_SLOW"));
 add\_opt(common\_arg(
 {"--yarn-beta-fast"}, "N",
 string\_format("YaRN: low correction dim or beta (default: %.2f)", (double)params.yarn\_beta\_fast),
 \[\](common\_params & params, const std::string & value) {
 params.yarn\_beta\_fast = std::stof(value);
 }
 ).set\_env("LLAMA\_ARG\_YARN\_BETA\_FAST"));
 add\_opt(common\_arg(
 {"-gan", "--grp-attn-n"}, "N",
 string\_format("group-attention factor (default: %d)", params.grp\_attn\_n),
 \[\](common\_params & params, int value) {
 params.grp\_attn\_n = value;
 }
 ).set\_env("LLAMA\_ARG\_GRP\_ATTN\_N").set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_PASSKEY}));
 add\_opt(common\_arg(
 {"-gaw", "--grp-attn-w"}, "N",
 string\_format("group-attention width (default: %d)", params.grp\_attn\_w),
 \[\](common\_params & params, int value) {
 params.grp\_attn\_w = value;
 }
 ).set\_env("LLAMA\_ARG\_GRP\_ATTN\_W").set\_examples({LLAMA\_EXAMPLE\_COMPLETION}));
 add\_opt(common\_arg(
 {"-kvo", "--kv-offload"},
 {"-nkvo", "--no-kv-offload"},
 string\_format("whether to enable KV cache offloading (default: %s)", params.no\_kv\_offload ? "disabled" : "enabled"),
 \[\](common\_params & params, bool value) {
 params.no\_kv\_offload = !value;
 }
 ).set\_env("LLAMA\_ARG\_KV\_OFFLOAD"));
 add\_opt(common\_arg(
 {"--repack"},
 {"-nr", "--no-repack"},
 string\_format("whether to enable weight repacking (default: %s)", params.no\_extra\_bufts ? "disabled" : "enabled"),
 \[\](common\_params & params, bool value) {
 params.no\_extra\_bufts = !value;
 }
 ).set\_env("LLAMA\_ARG\_REPACK"));
 add\_opt(common\_arg(
 {"--no-host"},
 "bypass host buffer allowing extra buffers to be used",
 \[\](common\_params & params) {
 params.no\_host = true;
 }
 ).set\_env("LLAMA\_ARG\_NO\_HOST"));
 add\_opt(common\_arg(
 {"-ctk", "--cache-type-k"}, "TYPE",
 string\_format(
 "KV cache data type for K\\n"
 "allowed values: %s\\n"
 "(default: %s)",
 get\_all\_kv\_cache\_types().c\_str(),
 ggml\_type\_name(params.cache\_type\_k)
 ),
 \[\](common\_params & params, const std::string & value) {
 params.cache\_type\_k = kv\_cache\_type\_from\_str(value);
 }
 ).set\_env("LLAMA\_ARG\_CACHE\_TYPE\_K"));
 add\_opt(common\_arg(
 {"-ctv", "--cache-type-v"}, "TYPE",
 string\_format(
 "KV cache data type for V\\n"
 "allowed values: %s\\n"
 "(default: %s)",
 get\_all\_kv\_cache\_types().c\_str(),
 ggml\_type\_name(params.cache\_type\_v)
 ),
 \[\](common\_params & params, const std::string & value) {
 params.cache\_type\_v = kv\_cache\_type\_from\_str(value);
 }
 ).set\_env("LLAMA\_ARG\_CACHE\_TYPE\_V"));
 add\_opt(common\_arg(
 {"--hellaswag"},
 "compute HellaSwag score over random tasks from datafile supplied with -f",
 \[\](common\_params & params) {
 params.hellaswag = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--hellaswag-tasks"}, "N",
 string\_format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag\_tasks),
 \[\](common\_params & params, int value) {
 params.hellaswag\_tasks = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--winogrande"},
 "compute Winogrande score over random tasks from datafile supplied with -f",
 \[\](common\_params & params) {
 params.winogrande = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--winogrande-tasks"}, "N",
 string\_format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande\_tasks),
 \[\](common\_params & params, int value) {
 params.winogrande\_tasks = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--multiple-choice"},
 "compute multiple choice score over random tasks from datafile supplied with -f",
 \[\](common\_params & params) {
 params.multiple\_choice = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--multiple-choice-tasks"}, "N",
 string\_format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple\_choice\_tasks),
 \[\](common\_params & params, int value) {
 params.multiple\_choice\_tasks = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--kl-divergence"},
 "computes KL-divergence to logits provided via --kl-divergence-base",
 \[\](common\_params & params) {
 params.kl\_divergence = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
 "set logits file",
 \[\](common\_params & params, const std::string & value) {
 params.logits\_file = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--ppl-stride"}, "N",
 string\_format("stride for perplexity calculation (default: %d)", params.ppl\_stride),
 \[\](common\_params & params, int value) {
 params.ppl\_stride = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"--ppl-output-type"}, "<0\|1>",
 string\_format("output type for perplexity calculation (default: %d)", params.ppl\_output\_type),
 \[\](common\_params & params, int value) {
 params.ppl\_output\_type = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PERPLEXITY}));
 add\_opt(common\_arg(
 {"-dt", "--defrag-thold"}, "N",
 string\_format("KV cache defragmentation threshold (DEPRECATED)"),
 \[\](common\_params & params, const std::string & value) {
 GGML\_UNUSED(params);
 GGML\_UNUSED(value);
 LOG\_WRN("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify\\n");
 }
 ).set\_env("LLAMA\_ARG\_DEFRAG\_THOLD"));
 if (ex == LLAMA\_EXAMPLE\_SERVER) {
 // this is to make sure this option appears in the server-specific section of the help message
 add\_opt(common\_arg(
 {"-np", "--parallel"}, "N",
 string\_format("number of server slots (default: %d, -1 = auto)", params.n\_parallel),
 \[\](common\_params & params, int value) {
 if (value == 0) {
 throw std::invalid\_argument("error: invalid value for n\_parallel\\n");
 }
 params.n\_parallel = value;
 }
 ).set\_env("LLAMA\_ARG\_N\_PARALLEL").set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 } else {
 add\_opt(common\_arg(
 {"-np", "--parallel"}, "N",
 string\_format("number of parallel sequences to decode (default: %d)", params.n\_parallel),
 \[\](common\_params & params, int value) {
 params.n\_parallel = value;
 }
 ).set\_env("LLAMA\_ARG\_N\_PARALLEL"));
 }
 add\_opt(common\_arg(
 {"-ns", "--sequences"}, "N",
 string\_format("number of sequences to decode (default: %d)", params.n\_sequences),
 \[\](common\_params & params, int value) {
 params.n\_sequences = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PARALLEL}));
 add\_opt(common\_arg(
 {"-cb", "--cont-batching"},
 {"-nocb", "--no-cont-batching"},
 string\_format("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont\_batching ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.cont\_batching = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_CONT\_BATCHING"));
 add\_opt(common\_arg(
 {"-mm", "--mmproj"}, "FILE",
 "path to a multimodal projector file. see tools/mtmd/README.md\\n"
 "note: if -hf is used, this argument can be omitted",
 \[\](common\_params & params, const std::string & value) {
 params.mmproj.path = value;
 }
 ).set\_examples(mmproj\_examples).set\_env("LLAMA\_ARG\_MMPROJ"));
 add\_opt(common\_arg(
 {"-mmu", "--mmproj-url"}, "URL",
 "URL to a multimodal projector file. see tools/mtmd/README.md",
 \[\](common\_params & params, const std::string & value) {
 params.mmproj.url = value;
 }
 ).set\_examples(mmproj\_examples).set\_env("LLAMA\_ARG\_MMPROJ\_URL"));
 add\_opt(common\_arg(
 {"--mmproj-auto"},
 {"--no-mmproj", "--no-mmproj-auto"},
 string\_format("whether to use multimodal projector file (if available), useful when using -hf (default: %s)", params.no\_mmproj ? "disabled" : "enabled"),
 \[\](common\_params & params, bool value) {
 params.no\_mmproj = !value;
 }
 ).set\_examples(mmproj\_examples).set\_env("LLAMA\_ARG\_MMPROJ\_AUTO"));
 add\_opt(common\_arg(
 {"--mmproj-offload"},
 {"--no-mmproj-offload"},
 string\_format("whether to enable GPU offloading for multimodal projector (default: %s)", params.mmproj\_use\_gpu ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.mmproj\_use\_gpu = value;
 }
 ).set\_examples(mmproj\_examples).set\_env("LLAMA\_ARG\_MMPROJ\_OFFLOAD"));
 add\_opt(common\_arg(
 {"--image", "--audio"}, "FILE",
 "path to an image or audio file. use with multimodal models, use comma-separated values for multiple files\\n",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 params.image.emplace\_back(item);
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_MTMD, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--image-min-tokens"}, "N",
 "minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
 \[\](common\_params & params, int value) {
 params.image\_min\_tokens = value;
 }
 ).set\_examples(mmproj\_examples).set\_env("LLAMA\_ARG\_IMAGE\_MIN\_TOKENS"));
 add\_opt(common\_arg(
 {"--image-max-tokens"}, "N",
 "maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
 \[\](common\_params & params, int value) {
 params.image\_max\_tokens = value;
 }
 ).set\_examples(mmproj\_examples).set\_env("LLAMA\_ARG\_IMAGE\_MAX\_TOKENS"));
 if (llama\_supports\_rpc()) {
 add\_opt(common\_arg(
 {"--rpc"}, "SERVERS",
 "comma-separated list of RPC servers (host:port)",
 \[\](common\_params & params, const std::string & value) {
 add\_rpc\_devices(value);
 GGML\_UNUSED(params);
 }
 ).set\_env("LLAMA\_ARG\_RPC"));
 }
 add\_opt(common\_arg(
 {"--mlock"},
 "force system to keep model in RAM rather than swapping or compressing",
 \[\](common\_params & params) {
 params.use\_mlock = true;
 }
 ).set\_env("LLAMA\_ARG\_MLOCK"));
 add\_opt(common\_arg(
 {"--mmap"},
 {"--no-mmap"},
 string\_format("whether to memory-map model. (if mmap disabled, slower load but may reduce pageouts if not using mlock) (default: %s)", params.use\_mmap ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.use\_mmap = value;
 }
 ).set\_env("LLAMA\_ARG\_MMAP"));
 add\_opt(common\_arg(
 {"-dio", "--direct-io"},
 {"-ndio", "--no-direct-io"},
 string\_format("use DirectIO if available. (default: %s)", params.use\_direct\_io ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.use\_direct\_io = value;
 }
 ).set\_env("LLAMA\_ARG\_DIO"));
 add\_opt(common\_arg(
 {"--numa"}, "TYPE",
 "attempt optimizations that help on some NUMA systems\\n"
 "\- distribute: spread execution evenly over all nodes\\n"
 "\- isolate: only spawn threads on CPUs on the node that execution started on\\n"
 "\- numactl: use the CPU map provided by numactl\\n"
 "if run without this previously, it is recommended to drop the system page cache before using this\\n"
 "see https://github.com/ggml-org/llama.cpp/issues/1437",
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "distribute" \|\| value == "") { params.numa = GGML\_NUMA\_STRATEGY\_DISTRIBUTE; }
 else if (value == "isolate") { params.numa = GGML\_NUMA\_STRATEGY\_ISOLATE; }
 else if (value == "numactl") { params.numa = GGML\_NUMA\_STRATEGY\_NUMACTL; }
 else { throw std::invalid\_argument("invalid value"); }
 }
 ).set\_env("LLAMA\_ARG\_NUMA"));
 add\_opt(common\_arg(
 {"-dev", "--device"}, "",
 "comma-separated list of devices to use for offloading (none = don't offload)\\n"
 "use --list-devices to see a list of available devices",
 \[\](common\_params & params, const std::string & value) {
 params.devices = parse\_device\_list(value);
 }
 ).set\_env("LLAMA\_ARG\_DEVICE"));
 add\_opt(common\_arg(
 {"--list-devices"},
 "print list of available devices and exit",
 \[\](common\_params &) {
 ggml\_backend\_load\_all();
 std::vector devices;
 for (size\_t i = 0; i < ggml\_backend\_dev\_count(); ++i) {
 auto \* dev = ggml\_backend\_dev\_get(i);
 if (ggml\_backend\_dev\_type(dev) != GGML\_BACKEND\_DEVICE\_TYPE\_CPU) {
 devices.push\_back(dev);
 }
 }
 printf("Available devices:\\n");
 for (auto \* dev : devices) {
 size\_t free, total;
 ggml\_backend\_dev\_memory(dev, &free, &total);
 printf(" %s: %s (%zu MiB, %zu MiB free)\\n", ggml\_backend\_dev\_name(dev), ggml\_backend\_dev\_description(dev), total / 1024 / 1024, free / 1024 / 1024);
 }
 exit(0);
 }
 ));
 add\_opt(common\_arg(
 {"-ot", "--override-tensor"}, "=,...",
 "override tensor buffer type", \[\](common\_params & params, const std::string & value) {
 parse\_tensor\_buffer\_overrides(value, params.tensor\_buft\_overrides);
 }
 ).set\_env("LLAMA\_ARG\_OVERRIDE\_TENSOR"));
 add\_opt(common\_arg(
 {"-cmoe", "--cpu-moe"},
 "keep all Mixture of Experts (MoE) weights in the CPU",
 \[\](common\_params & params) {
 params.tensor\_buft\_overrides.push\_back(llm\_ffn\_exps\_cpu\_override());
 }
 ).set\_env("LLAMA\_ARG\_CPU\_MOE"));
 add\_opt(common\_arg(
 {"-ncmoe", "--n-cpu-moe"}, "N",
 "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
 \[\](common\_params & params, int value) {
 if (value < 0) {
 throw std::invalid\_argument("invalid value");
 }
 for (int i = 0; i < value; ++i) {
 // keep strings alive and avoid leaking memory by storing them in a static vector
 static std::list buft\_overrides;
 buft\_overrides.push\_back(llm\_ffn\_exps\_block\_regex(i));
 params.tensor\_buft\_overrides.push\_back({buft\_overrides.back().c\_str(), ggml\_backend\_cpu\_buffer\_type()});
 }
 }
 ).set\_env("LLAMA\_ARG\_N\_CPU\_MOE"));
 GGML\_ASSERT(params.n\_gpu\_layers < 0); // string\_format would need to be extended for a default >= 0
 add\_opt(common\_arg(
 {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
 string\_format("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)", params.n\_gpu\_layers == -1 ? "auto" : "all"),
 \[\](common\_params & params, const std::string & value) {
 if (value == "auto") {
 params.n\_gpu\_layers = -1;
 } else if (value == "all") {
 params.n\_gpu\_layers = -2;
 } else {
 params.n\_gpu\_layers = std::stoi(value);
 }
 if (!llama\_supports\_gpu\_offload()) {
 fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\\n");
 fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\\n");
 fprintf(stderr, "warning: consult docs/build.md for compilation instructions\\n");
 }
 }
 ).set\_env("LLAMA\_ARG\_N\_GPU\_LAYERS"));
 add\_opt(common\_arg(
 {"-sm", "--split-mode"}, "{none,layer,row,tensor}",
 "how to split the model across multiple GPUs, one of:\\n"
 "\- none: use one GPU only\\n"
 "\- layer (default): split layers and KV across GPUs (pipelined)\\n"
 "\- row: split weight across GPUs by rows (parallelized)\\n"
 "\- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",
 \[\](common\_params & params, const std::string & value) {
 if (value == "none") {
 params.split\_mode = LLAMA\_SPLIT\_MODE\_NONE;
 } else if (value == "layer") {
 params.split\_mode = LLAMA\_SPLIT\_MODE\_LAYER;
 } else if (value == "row") {
 params.split\_mode = LLAMA\_SPLIT\_MODE\_ROW;
 } else if (value == "tensor") {
 params.split\_mode = LLAMA\_SPLIT\_MODE\_TENSOR;
 } else {
 throw std::invalid\_argument("invalid value");
 }
 if (!llama\_supports\_gpu\_offload()) {
 fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\\n");
 }
 }
 ).set\_env("LLAMA\_ARG\_SPLIT\_MODE"));
 add\_opt(common\_arg(
 {"-ts", "--tensor-split"}, "N0,N1,N2,...",
 "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
 \[\](common\_params & params, const std::string & value) {
 std::string arg\_next = value;

 // split string by , and /
 const std::regex regex{ R"(\[,/\]+)" };
 std::sregex\_token\_iterator it{ arg\_next.begin(), arg\_next.end(), regex, -1 };
 std::vector split\_arg{ it, {} };
 if (split\_arg.size() >= llama\_max\_devices()) {
 throw std::invalid\_argument(
 string\_format("got %zu input configs, but system only has %zu devices", split\_arg.size(), llama\_max\_devices())
 );
 }
 for (size\_t i = 0; i < llama\_max\_devices(); ++i) {
 if (i < split\_arg.size()) {
 params.tensor\_split\[i\] = std::stof(split\_arg\[i\]);
 } else {
 params.tensor\_split\[i\] = 0.0f;
 }
 }
 if (!llama\_supports\_gpu\_offload()) {
 fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\\n");
 }
 }
 ).set\_env("LLAMA\_ARG\_TENSOR\_SPLIT"));
 add\_opt(common\_arg(
 {"-mg", "--main-gpu"}, "INDEX",
 string\_format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main\_gpu),
 \[\](common\_params & params, int value) {
 params.main\_gpu = value;
 if (!llama\_supports\_gpu\_offload()) {
 fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\\n");
 }
 }
 ).set\_env("LLAMA\_ARG\_MAIN\_GPU"));
 add\_opt(common\_arg(
 { "-fit", "--fit" }, "\[on\|off\]",
 string\_format("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')", params.fit\_params ? "on" : "off"),
 \[\](common\_params & params, const std::string & value) {
 if (is\_truthy(value)) {
 params.fit\_params = true;
 } else if (is\_falsey(value)) {
 params.fit\_params = false;
 } else {
 throw std::runtime\_error(
 string\_format("error: unknown value for --fit: '%s'\\n", value.c\_str()));
 }
 }
 ).set\_env("LLAMA\_ARG\_FIT"));
 add\_opt(common\_arg(
 { "-fitp", "--fit-print" }, "\[on\|off\]",
 string\_format("print the estimated required memory ('on' or 'off', default: '%s')", params.fit\_params\_print ? "on" : "off"),
 \[\](common\_params & params, const std::string & value) {
 if (is\_truthy(value)) {
 params.fit\_params\_print = true;
 } else if (is\_falsey(value)) {
 params.fit\_params\_print = false;
 } else {
 throw std::runtime\_error(
 string\_format("error: unknown value for --fit-print: '%s'\\n", value.c\_str()));
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_FIT\_PARAMS}).set\_env("LLAMA\_ARG\_FIT\_ESTIMATE"));
 add\_opt(common\_arg(
 { "-fitt", "--fit-target" }, "MiB0,MiB1,MiB2,...",
 string\_format("target margin per device for --fit, comma-separated list of values, "
 "single value is broadcast across all devices, default: %zu", params.fit\_params\_target\[0\]/(1024\*1024)),
 \[\](common\_params & params, const std::string & value) {
 std::string arg\_next = value;

 // split string by , and /
 const std::regex regex{ R"(\[,/\]+)" };
 std::sregex\_token\_iterator it{ arg\_next.begin(), arg\_next.end(), regex, -1 };
 std::vector split\_arg{ it, {} };
 if (split\_arg.size() >= llama\_max\_devices()) {
 throw std::invalid\_argument(
 string\_format("got %zu input configs, but system only has %zu devices", split\_arg.size(), llama\_max\_devices())
 );
 }
 if (split\_arg.size() == 1) {
 std::fill(params.fit\_params\_target.begin(), params.fit\_params\_target.end(), std::stoull(split\_arg\[0\]) \* 1024\*1024);
 return;
 }
 for (size\_t i = 0; i < split\_arg.size(); i++) {
 params.fit\_params\_target\[i\] = std::stoull(split\_arg\[i\]) \* 1024\*1024;
 }
 }
 ).set\_env("LLAMA\_ARG\_FIT\_TARGET"));
 add\_opt(common\_arg(
 { "-fitc", "--fit-ctx" }, "N",
 string\_format("minimum ctx size that can be set by --fit option, default: %" PRIu32, params.fit\_params\_min\_ctx),
 \[\](common\_params & params, int value) {
 params.fit\_params\_min\_ctx = value;
 }
 ).set\_env("LLAMA\_ARG\_FIT\_CTX"));
 add\_opt(common\_arg(
 {"--check-tensors"},
 string\_format("check model tensor data for invalid values (default: %s)", params.check\_tensors ? "true" : "false"),
 \[\](common\_params & params) {
 params.check\_tensors = true;
 }
 ));
 add\_opt(common\_arg(
 {"--override-kv"}, "KEY=TYPE:VALUE,...",
 "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\\n"
 "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add\_bos\_token=bool:false,tokenizer.ggml.add\_eos\_token=bool:false",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 if (!string\_parse\_kv\_override(item.c\_str(), params.kv\_overrides)) {
 throw std::runtime\_error(string\_format("error: Invalid type for KV override: %s\\n", item.c\_str()));
 }
 }
 }
 ));
 add\_opt(common\_arg(
 {"--op-offload"},
 {"--no-op-offload"},
 string\_format("whether to offload host tensor operations to device (default: %s)", params.no\_op\_offload ? "false" : "true"),
 \[\](common\_params & params, bool value) {
 params.no\_op\_offload = !value;
 }
 ));
 add\_opt(common\_arg(
 {"--lora"}, "FNAME",
 "path to LoRA adapter (use comma-separated values to load multiple adapters)",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 params.lora\_adapters.push\_back({ item, 1.0, "", "", nullptr });
 }
 }
 // we define this arg on both COMMON and EXPORT\_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
 ).set\_examples({LLAMA\_EXAMPLE\_COMMON, LLAMA\_EXAMPLE\_EXPORT\_LORA}));
 add\_opt(common\_arg(
 {"--lora-scaled"}, "FNAME:SCALE,...",
 "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\\n"
 "note: use comma-separated values",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 auto parts = string\_split(item, ':');
 if (parts.size() != 2) {
 throw std::invalid\_argument("lora-scaled format: FNAME:SCALE");
 }
 params.lora\_adapters.push\_back({ parts\[0\], std::stof(parts\[1\]), "", "", nullptr });
 }
 }
 // we define this arg on both COMMON and EXPORT\_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
 ).set\_examples({LLAMA\_EXAMPLE\_COMMON, LLAMA\_EXAMPLE\_EXPORT\_LORA}));
 add\_opt(common\_arg(
 {"--control-vector"}, "FNAME",
 "add a control vector\\nnote: use comma-separated values to add multiple control vectors",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 params.control\_vectors.push\_back({ 1.0f, item, });
 }
 }
 ));
 add\_opt(common\_arg(
 {"--control-vector-scaled"}, "FNAME:SCALE,...",
 "add a control vector with user defined scaling SCALE\\n"
 "note: use comma-separated values (format: FNAME:SCALE,...)",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 auto parts = string\_split(item, ':');
 if (parts.size() != 2) {
 throw std::invalid\_argument("control-vector-scaled format: FNAME:SCALE");
 }
 params.control\_vectors.push\_back({ std::stof(parts\[1\]), parts\[0\] });
 }
 }
 ));
 add\_opt(common\_arg(
 {"--control-vector-layer-range"}, "START", "END",
 "layer range to apply the control vector(s) to, start and end inclusive",
 \[\](common\_params & params, const std::string & start, const std::string & end) {
 params.control\_vector\_layer\_start = std::stoi(start);
 params.control\_vector\_layer\_end = std::stoi(end);
 }
 ));
 add\_opt(common\_arg(
 {"-a", "--alias"}, "STRING",
 "set model name aliases, comma-separated (to be used by API)",
 \[\](common\_params & params, const std::string & value) {
 for (auto & alias : string\_split(value, ',')) {
 alias = string\_strip(alias);
 if (!alias.empty()) {
 params.model\_alias.insert(alias);
 }
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_ALIAS"));
 add\_opt(common\_arg(
 {"--tags"}, "STRING",
 "set model tags, comma-separated (informational, not used for routing)",
 \[\](common\_params & params, const std::string & value) {
 for (auto & tag : string\_split(value, ',')) {
 tag = string\_strip(tag);
 if (!tag.empty()) {
 params.model\_tags.insert(tag);
 }
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_TAGS"));
 add\_opt(common\_arg(
 {"-m", "--model"}, "FNAME",
 ex == LLAMA\_EXAMPLE\_EXPORT\_LORA
 ? "model path from which to load base model"
 : "model path to load",
 \[\](common\_params & params, const std::string & value) {
 params.model.path = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMMON, LLAMA\_EXAMPLE\_EXPORT\_LORA}).set\_env("LLAMA\_ARG\_MODEL"));
 add\_opt(common\_arg(
 {"-mu", "--model-url"}, "MODEL\_URL",
 "model download url (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.model.url = value;
 }
 ).set\_env("LLAMA\_ARG\_MODEL\_URL"));
 add\_opt(common\_arg(
 { "-dr", "--docker-repo" }, "\[/\]\[:quant\]",
 "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\\n"
 "example: gemma3\\n"
 "(default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.model.docker\_repo = value;
 }
 ).set\_env("LLAMA\_ARG\_DOCKER\_REPO"));
 add\_opt(common\_arg(
 {"-hf", "-hfr", "--hf-repo"}, "/\[:quant\]",
 "Hugging Face model repository; quant is optional, case-insensitive, default to Q4\_K\_M, or falls back to the first file in the repo if Q4\_K\_M doesn't exist.\\n"
 "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\\n"
 "example: ggml-org/GLM-4.7-Flash-GGUF:Q4\_K\_M\\n"
 "(default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.model.hf\_repo = value;
 }
 ).set\_env("LLAMA\_ARG\_HF\_REPO"));
 add\_opt(common\_arg(
 {"-hff", "--hf-file"}, "FILE",
 "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.model.hf\_file = value;
 }
 ).set\_env("LLAMA\_ARG\_HF\_FILE"));
 add\_opt(common\_arg(
 {"-hfv", "-hfrv", "--hf-repo-v"}, "/\[:quant\]",
 "Hugging Face model repository for the vocoder model (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.vocoder.model.hf\_repo = value;
 }
 ).set\_env("LLAMA\_ARG\_HF\_REPO\_V"));
 add\_opt(common\_arg(
 {"-hffv", "--hf-file-v"}, "FILE",
 "Hugging Face model file for the vocoder model (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.vocoder.model.hf\_file = value;
 }
 ).set\_env("LLAMA\_ARG\_HF\_FILE\_V"));
 add\_opt(common\_arg(
 {"-hft", "--hf-token"}, "TOKEN",
 "Hugging Face access token (default: value from HF\_TOKEN environment variable)",
 \[\](common\_params & params, const std::string & value) {
 params.hf\_token = value;
 }
 ).set\_env("HF\_TOKEN"));
 add\_opt(common\_arg(
 {"--context-file"}, "FNAME",
 "file to load context from (use comma-separated values to specify multiple files)",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & item : parse\_csv\_row(value)) {
 std::ifstream file(item, std::ios::binary);
 if (!file) {
 throw std::runtime\_error(string\_format("error: failed to open file '%s'\\n", item.c\_str()));
 }
 params.context\_files.push\_back(item);
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_RETRIEVAL}));
 add\_opt(common\_arg(
 {"--chunk-size"}, "N",
 string\_format("minimum length of embedded text chunks (default: %d)", params.chunk\_size),
 \[\](common\_params & params, int value) {
 params.chunk\_size = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_RETRIEVAL}));
 add\_opt(common\_arg(
 {"--chunk-separator"}, "STRING",
 string\_format("separator between chunks (default: '%s')", params.chunk\_separator.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.chunk\_separator = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_RETRIEVAL}));
 add\_opt(common\_arg(
 {"--junk"}, "N",
 string\_format("number of times to repeat the junk text (default: %d)", params.n\_junk),
 \[\](common\_params & params, int value) {
 params.n\_junk = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PASSKEY, LLAMA\_EXAMPLE\_PARALLEL}));
 add\_opt(common\_arg(
 {"--pos"}, "N",
 string\_format("position of the passkey in the junk text (default: %d)", params.i\_pos),
 \[\](common\_params & params, int value) {
 params.i\_pos = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_PASSKEY}));
 add\_opt(common\_arg(
 {"-o", "--output", "--output-file"}, "FNAME",
 string\_format("output file (default: '%s')", params.out\_file.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.out\_file = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX, LLAMA\_EXAMPLE\_CVECTOR\_GENERATOR, LLAMA\_EXAMPLE\_EXPORT\_LORA, LLAMA\_EXAMPLE\_TTS, LLAMA\_EXAMPLE\_FINETUNE,
 LLAMA\_EXAMPLE\_RESULTS, LLAMA\_EXAMPLE\_EXPORT\_GRAPH\_OPS}));
 add\_opt(common\_arg(
 {"-ofreq", "--output-frequency"}, "N",
 string\_format("output the imatrix every N iterations (default: %d)", params.n\_out\_freq),
 \[\](common\_params & params, int value) {
 params.n\_out\_freq = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--output-format"}, "{gguf,dat}",
 string\_format("output format for imatrix file (default: %s)", params.imat\_dat > 0 ? "dat" : "gguf"),
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "gguf") { params.imat\_dat = -1; }
 else if (value == "dat") { params.imat\_dat = 1; }
 else { throw std::invalid\_argument("invalid output format"); }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--save-frequency"}, "N",
 string\_format("save an imatrix copy every N iterations (default: %d)", params.n\_save\_freq),
 \[\](common\_params & params, int value) {
 params.n\_save\_freq = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--process-output"},
 string\_format("collect data for the output tensor (default: %s)", params.process\_output ? "true" : "false"),
 \[\](common\_params & params) {
 params.process\_output = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--ppl"},
 {"--no-ppl"},
 string\_format("whether to compute perplexity (default: %s)", params.compute\_ppl ? "true" : "false"),
 \[\](common\_params & params, bool value) {
 params.compute\_ppl = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--chunk", "--from-chunk"}, "N",
 string\_format("start processing the input from chunk N (default: %d)", params.i\_chunk),
 \[\](common\_params & params, int value) {
 params.i\_chunk = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--show-statistics"},
 string\_format("show imatrix statistics and then exit (default: %s)", params.show\_statistics ? "true" : "false"),
 \[\](common\_params & params) {
 params.show\_statistics = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"--parse-special"},
 string\_format("parse special tokens (chat, tool, etc) (default: %s)", params.parse\_special ? "true" : "false"),
 \[\](common\_params & params) {
 params.parse\_special = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_IMATRIX}));
 add\_opt(common\_arg(
 {"-pps"},
 string\_format("is the prompt shared across parallel sequences (default: %s)", params.is\_pp\_shared ? "true" : "false"),
 \[\](common\_params & params) {
 params.is\_pp\_shared = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_BENCH, LLAMA\_EXAMPLE\_PARALLEL}));
 add\_opt(common\_arg(
 {"-tgs"},
 string\_format("is the text generation separated across the different sequences (default: %s)", params.is\_tg\_separate ? "true" : "false"),
 \[\](common\_params & params) {
 params.is\_tg\_separate = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_BENCH, LLAMA\_EXAMPLE\_PARALLEL}));
 add\_opt(common\_arg(
 {"-npp"}, "n0,n1,...",
 "number of prompt tokens",
 \[\](common\_params & params, const std::string & value) {
 auto p = string\_split(value, ',');
 params.n\_pp.insert(params.n\_pp.end(), p.begin(), p.end());
 }
 ).set\_examples({LLAMA\_EXAMPLE\_BENCH}));
 add\_opt(common\_arg(
 {"-ntg"}, "n0,n1,...",
 "number of text generation tokens",
 \[\](common\_params & params, const std::string & value) {
 auto p = string\_split(value, ',');
 params.n\_tg.insert(params.n\_tg.end(), p.begin(), p.end());
 }
 ).set\_examples({LLAMA\_EXAMPLE\_BENCH}));
 add\_opt(common\_arg(
 {"-npl"}, "n0,n1,...",
 "number of parallel prompts",
 \[\](common\_params & params, const std::string & value) {
 auto p = string\_split(value, ',');
 params.n\_pl.insert(params.n\_pl.end(), p.begin(), p.end());
 }
 ).set\_examples({LLAMA\_EXAMPLE\_BENCH}));
 add\_opt(common\_arg(
 {"--embd-normalize"}, "N",
 string\_format("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd\_normalize),
 \[\](common\_params & params, int value) {
 params.embd\_normalize = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_DEBUG}));
 add\_opt(common\_arg(
 {"--embd-output-format"}, "FORMAT",
 "empty = default, \\"array\\" = \[\[\],\[\]...\], \\"json\\" = openai style, \\"json+\\" = same \\"json\\" + cosine similarity matrix, \\"raw\\" = plain whitespace-delimited output (one embedding per line)",
 \[\](common\_params & params, const std::string & value) {
 params.embd\_out = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING}));
 add\_opt(common\_arg(
 {"--embd-separator"}, "STRING",
 "separator of embeddings (default \\\n) for example \\"<#sep#>\\"",
 \[\](common\_params & params, const std::string & value) {
 params.embd\_sep = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING}));
 add\_opt(common\_arg(
 {"--cls-separator"}, "STRING",
 "separator of classification sequences (default \\\t) for example \\"<#seq#>\\"",
 \[\](common\_params & params, const std::string & value) {
 params.cls\_sep = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING}));
 add\_opt(common\_arg(
 {"--host"}, "HOST",
 string\_format("ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: %s)", params.hostname.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.hostname = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_HOST"));
 add\_opt(common\_arg(
 {"--port"}, "PORT",
 string\_format("port to listen (default: %d)", params.port),
 \[\](common\_params & params, int value) {
 params.port = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_PORT"));
 add\_opt(common\_arg(
 {"--reuse-port"},
 string\_format("allow multiple sockets to bind to the same port (default: %s)", params.reuse\_port ? "enabled" : "disabled"),
 \[\](common\_params & params) {
 params.reuse\_port = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_REUSE\_PORT"));
 add\_opt(common\_arg(
 {"--path"}, "PATH",
 string\_format("path to serve static files from (default: %s)", params.public\_path.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.public\_path = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_STATIC\_PATH"));
 add\_opt(common\_arg(
 {"--api-prefix"}, "PREFIX",
 string\_format("prefix path the server serves from, without the trailing slash (default: %s)", params.api\_prefix.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.api\_prefix = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_API\_PREFIX"));
 // Deprecated: use --ui-config instead (kept for backward compat)
 add\_opt(common\_arg(
 {"--webui-config"}, "JSON",
 "\[DEPRECATED: use --ui-config\] JSON that provides default WebUI settings (overrides WebUI defaults)",
 \[\](common\_params & params, const std::string & value) {
 params.ui\_config\_json = value;
 params.webui\_config\_json = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_WEBUI\_CONFIG"));

 add\_opt(common\_arg(
 {"--ui-config"}, "JSON",
 "JSON that provides default UI settings (overrides UI defaults)",
 \[\](common\_params & params, const std::string & value) {
 params.ui\_config\_json = value;
 params.webui\_config\_json = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_UI\_CONFIG"));

 // Deprecated: use --ui-config-file instead (kept for backward compat)
 add\_opt(common\_arg(
 {"--webui-config-file"}, "PATH",
 "\[DEPRECATED: use --ui-config-file\] JSON file that provides default WebUI settings (overrides WebUI defaults)",
 \[\](common\_params & params, const std::string & value) {
 params.ui\_config\_json = read\_file(value);
 params.webui\_config\_json = params.ui\_config\_json;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_WEBUI\_CONFIG\_FILE"));

 add\_opt(common\_arg(
 {"--ui-config-file"}, "PATH",
 "JSON file that provides default UI settings (overrides UI defaults)",
 \[\](common\_params & params, const std::string & value) {
 params.ui\_config\_json = read\_file(value);
 params.webui\_config\_json = params.ui\_config\_json;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_UI\_CONFIG\_FILE"));

 // Deprecated: use --ui-mcp-proxy instead (kept for backward compat)
 add\_opt(common\_arg(
 {"--webui-mcp-proxy"},
 {"--no-webui-mcp-proxy"},
 "\[DEPRECATED: use --ui-mcp-proxy/--no-ui-mcp-proxy\] experimental: whether to enable MCP CORS proxy",
 \[\](common\_params & params, bool value) {
 params.ui\_mcp\_proxy = value;
 params.webui\_mcp\_proxy = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_WEBUI\_MCP\_PROXY"));

 add\_opt(common\_arg(
 {"--ui-mcp-proxy"},
 {"--no-ui-mcp-proxy"},
 "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments (default: disabled)",
 \[\](common\_params & params, bool value) {
 params.ui\_mcp\_proxy = value;
 params.webui\_mcp\_proxy = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_UI\_MCP\_PROXY"));
 add\_opt(common\_arg(
 {"--tools"}, "TOOL1,TOOL2,...",
 "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\\n"
 "specify \\"all\\" to enable all tools\\n"
 "available tools: read\_file, file\_glob\_search, grep\_search, exec\_shell\_command, write\_file, edit\_file, apply\_diff, get\_datetime",
 \[\](common\_params & params, const std::string & value) {
 params.server\_tools = parse\_csv\_row(value);
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_TOOLS"));
 // Deprecated: use --ui/--no-ui instead (kept for backward compat)
 add\_opt(common\_arg(
 {"--webui"},
 {"--no-webui"},
 "\[DEPRECATED: use --ui/--no-ui\] whether to enable the Web UI",
 \[\](common\_params & params, bool value) {
 params.ui = value;
 params.webui = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_WEBUI"));

 add\_opt(common\_arg(
 {"--ui"},
 {"--no-ui"},
 string\_format("whether to enable the Web UI (default: %s)", params.ui ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.ui = value;
 params.webui = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_UI"));
 add\_opt(common\_arg(
 {"--embedding", "--embeddings"},
 string\_format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
 \[\](common\_params & params) {
 params.embedding = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_DEBUG}).set\_env("LLAMA\_ARG\_EMBEDDINGS"));
 add\_opt(common\_arg(
 {"--rerank", "--reranking"},
 string\_format("enable reranking endpoint on server (default: %s)", "disabled"),
 \[\](common\_params & params) {
 params.embedding = true;
 params.pooling\_type = LLAMA\_POOLING\_TYPE\_RANK;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_RERANKING"));
 add\_opt(common\_arg(
 {"--api-key"}, "KEY",
 "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)",
 \[\](common\_params & params, const std::string & value) {
 for (const auto & key : parse\_csv\_row(value)) {
 if (!key.empty()) {
 params.api\_keys.push\_back(key);
 }
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_API\_KEY"));
 add\_opt(common\_arg(
 {"--api-key-file"}, "FNAME",
 "path to file containing API keys (default: none)",
 \[\](common\_params & params, const std::string & value) {
 std::ifstream key\_file(value);
 if (!key\_file) {
 throw std::runtime\_error(string\_format("error: failed to open file '%s'\\n", value.c\_str()));
 }
 std::string key;
 while (std::getline(key\_file, key)) {
 if (!key.empty()) {
 params.api\_keys.push\_back(key);
 }
 }
 key\_file.close();
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--ssl-key-file"}, "FNAME",
 "path to file a PEM-encoded SSL private key",
 \[\](common\_params & params, const std::string & value) {
 params.ssl\_file\_key = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_SSL\_KEY\_FILE"));
 add\_opt(common\_arg(
 {"--ssl-cert-file"}, "FNAME",
 "path to file a PEM-encoded SSL certificate",
 \[\](common\_params & params, const std::string & value) {
 params.ssl\_file\_cert = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_SSL\_CERT\_FILE"));
 add\_opt(common\_arg(
 {"--chat-template-kwargs"}, "STRING",
 "sets additional params for the json template parser, must be a valid json object string, e.g. '{\\"key1\\":\\"value1\\",\\"key2\\":\\"value2\\"}'",
 \[\](common\_params & params, const std::string & value) {
 auto parsed = json::parse(value);
 for (const auto & item : parsed.items()) {
 if (item.key() == "enable\_thinking") {
 LOG\_WRN("Setting 'enable\_thinking' via --chat-template-kwargs is deprecated. "
 "Use --reasoning on / --reasoning off instead.\\n");
 }
 params.default\_template\_kwargs\[item.key()\] = item.value().dump();
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_CHAT\_TEMPLATE\_KWARGS"));
 add\_opt(common\_arg(
 {"-to", "--timeout"}, "N",
 string\_format("server read/write timeout in seconds (default: %d)", params.timeout\_read),
 \[\](common\_params & params, int value) {
 params.timeout\_read = value;
 params.timeout\_write = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_TIMEOUT"));
 add\_opt(common\_arg(
 {"--threads-http"}, "N",
 string\_format("number of threads used to process HTTP requests (default: %d)", params.n\_threads\_http),
 \[\](common\_params & params, int value) {
 params.n\_threads\_http = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_THREADS\_HTTP"));
 add\_opt(common\_arg(
 {"--cache-prompt"},
 {"--no-cache-prompt"},
 string\_format("whether to enable prompt caching (default: %s)", params.cache\_prompt ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.cache\_prompt = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_CACHE\_PROMPT"));
 add\_opt(common\_arg(
 {"--cache-reuse"}, "N",
 string\_format(
 "min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: %d)\\n"
 "\[(card)\](https://ggml.ai/f0.png)", params.n\_cache\_reuse
 ),
 \[\](common\_params & params, int value) {
 params.n\_cache\_reuse = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_CACHE\_REUSE"));
 add\_opt(common\_arg(
 {"--metrics"},
 string\_format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint\_metrics ? "enabled" : "disabled"),
 \[\](common\_params & params) {
 params.endpoint\_metrics = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_ENDPOINT\_METRICS"));
 add\_opt(common\_arg(
 {"--props"},
 string\_format("enable changing global properties via POST /props (default: %s)", params.endpoint\_props ? "enabled" : "disabled"),
 \[\](common\_params & params) {
 params.endpoint\_props = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_ENDPOINT\_PROPS"));
 add\_opt(common\_arg(
 {"--slots"},
 {"--no-slots"},
 string\_format("expose slots monitoring endpoint (default: %s)", params.endpoint\_slots ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.endpoint\_slots = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_ENDPOINT\_SLOTS"));
 add\_opt(common\_arg(
 {"--slot-save-path"}, "PATH",
 "path to save slot kv cache (default: disabled)",
 \[\](common\_params & params, const std::string & value) {
 params.slot\_save\_path = value;
 if (!fs\_is\_directory(params.slot\_save\_path)) {
 throw std::invalid\_argument("not a directory: " + value);
 }
 // if doesn't end with DIRECTORY\_SEPARATOR, add it
 if (!params.slot\_save\_path.empty() && params.slot\_save\_path\[params.slot\_save\_path.size() - 1\] != DIRECTORY\_SEPARATOR) {
 params.slot\_save\_path += DIRECTORY\_SEPARATOR;
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--media-path"}, "PATH",
 "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)",
 \[\](common\_params & params, const std::string & value) {
 params.media\_path = value;
 if (!fs\_is\_directory(params.media\_path)) {
 throw std::invalid\_argument("not a directory: " + value);
 }
 // if doesn't end with DIRECTORY\_SEPARATOR, add it
 if (!params.media\_path.empty() && params.media\_path\[params.media\_path.size() - 1\] != DIRECTORY\_SEPARATOR) {
 params.media\_path += DIRECTORY\_SEPARATOR;
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--models-dir"}, "PATH",
 "directory containing models for the router server (default: disabled)",
 \[\](common\_params & params, const std::string & value) {
 params.models\_dir = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_MODELS\_DIR"));
 add\_opt(common\_arg(
 {"--models-preset"}, "PATH",
 "path to INI file containing model presets for the router server (default: disabled)",
 \[\](common\_params & params, const std::string & value) {
 params.models\_preset = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_MODELS\_PRESET"));
 add\_opt(common\_arg(
 {"--models-max"}, "N",
 string\_format("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)", params.models\_max),
 \[\](common\_params & params, int value) {
 params.models\_max = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_MODELS\_MAX"));
 add\_opt(common\_arg(
 {"--models-autoload"},
 {"--no-models-autoload"},
 string\_format("for router server, whether to automatically load models (default: %s)", params.models\_autoload ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.models\_autoload = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_MODELS\_AUTOLOAD"));
 add\_opt(common\_arg(
 {"--jinja"},
 {"--no-jinja"},
 string\_format("whether to use jinja template engine for chat (default: %s)", params.use\_jinja ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.use\_jinja = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_MTMD}).set\_env("LLAMA\_ARG\_JINJA"));
 add\_opt(common\_arg(
 {"--reasoning-format"}, "FORMAT",
 "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\\n"
 "\- none: leaves thoughts unparsed in \`message.content\`\\n"
 "\- deepseek: puts thoughts in \`message.reasoning\_content\`\\n"
 "\- deepseek-legacy: keeps \`\` tags in \`message.content\` while also populating \`message.reasoning\_content\`\\n"
 "(default: auto)",
 \[\](common\_params & params, const std::string & value) {
 params.reasoning\_format = common\_reasoning\_format\_from\_name(value);
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_THINK"));
 add\_opt(common\_arg(
 {"-rea", "--reasoning"}, "\[on\|off\|auto\]",
 "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
 \[\](common\_params & params, const std::string & value) {
 if (is\_truthy(value)) {
 params.enable\_reasoning = 1;
 params.default\_template\_kwargs\["enable\_thinking"\] = "true";
 } else if (is\_falsey(value)) {
 params.enable\_reasoning = 0;
 params.default\_template\_kwargs\["enable\_thinking"\] = "false";
 } else if (is\_autoy(value)) {
 params.enable\_reasoning = -1;
 } else {
 throw std::invalid\_argument(
 string\_format("error: unknown value for --reasoning: '%s'\\n", value.c\_str()));
 }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_REASONING"));
 add\_opt(common\_arg(
 {"--reasoning-budget"}, "N",
 "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
 \[\](common\_params & params, int value) {
 if (value < -1) { throw std::invalid\_argument("invalid value"); }
 params.sampling.reasoning\_budget\_tokens = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_THINK\_BUDGET"));
 add\_opt(common\_arg(
 {"--reasoning-budget-message"}, "MESSAGE",
 "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
 \[\](common\_params & params, const std::string & value) {
 params.sampling.reasoning\_budget\_message = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_THINK\_BUDGET\_MESSAGE"));
 add\_opt(common\_arg(
 {"--chat-template"}, "JINJA\_TEMPLATE",
 string\_format(
 "set custom jinja chat template (default: template taken from model's metadata)\\n"
 "if suffix/prefix are specified, template will be disabled\\n"
 "only commonly used templates are accepted (unless --jinja is set before this flag):\\n"
 "list of built-in templates:\\n%s", list\_builtin\_chat\_templates().c\_str()
 ),
 \[\](common\_params & params, const std::string & value) {
 params.chat\_template = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_MTMD}).set\_env("LLAMA\_ARG\_CHAT\_TEMPLATE"));
 add\_opt(common\_arg(
 {"--chat-template-file"}, "JINJA\_TEMPLATE\_FILE",
 string\_format(
 "set custom jinja chat template file (default: template taken from model's metadata)\\n"
 "if suffix/prefix are specified, template will be disabled\\n"
 "only commonly used templates are accepted (unless --jinja is set before this flag):\\n"
 "list of built-in templates:\\n%s", list\_builtin\_chat\_templates().c\_str()
 ),
 \[\](common\_params & params, const std::string & value) {
 params.chat\_template = read\_file(value);
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_CHAT\_TEMPLATE\_FILE"));
 add\_opt(common\_arg(
 {"--skip-chat-parsing"},
 {"--no-skip-chat-parsing"},
 string\_format(
 "force a pure content parser, even if a Jinja template is specified; model will output everything "
 "in the content section, including any reasoning and/or tool calls (default: disabled)"
 ),
 \[\](common\_params & params, bool value) {
 params.force\_pure\_content\_parser = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI, LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_SKIP\_CHAT\_PARSING"));
 add\_opt(common\_arg(
 {"--prefill-assistant"},
 {"--no-prefill-assistant"},
 string\_format(
 "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\\n"
 "when this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\\n"
 ),
 \[\](common\_params & params, bool value) {
 params.prefill\_assistant = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}).set\_env("LLAMA\_ARG\_PREFILL\_ASSISTANT"));
 add\_opt(common\_arg(
 {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
 string\_format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\\n", params.slot\_prompt\_similarity),
 \[\](common\_params & params, const std::string & value) {
 params.slot\_prompt\_similarity = std::stof(value);
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--lora-init-without-apply"},
 string\_format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora\_init\_without\_apply ? "enabled" : "disabled"),
 \[\](common\_params & params) {
 params.lora\_init\_without\_apply = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--sleep-idle-seconds"}, "SECONDS",
 string\_format("number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)", params.sleep\_idle\_seconds),
 \[\](common\_params & params, int value) {
 if (value == 0 \|\| value < -1) {
 throw std::invalid\_argument("invalid value: cannot be 0 or less than -1");
 }
 params.sleep\_idle\_seconds = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--simple-io"},
 "use basic IO for better compatibility in subprocesses and limited consoles",
 \[\](common\_params & params) {
 params.simple\_io = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_COMPLETION, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--positive-file"}, "FNAME",
 string\_format("positive prompts file, one prompt per line (default: '%s')", params.cvector\_positive\_file.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.cvector\_positive\_file = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_CVECTOR\_GENERATOR}));
 add\_opt(common\_arg(
 {"--negative-file"}, "FNAME",
 string\_format("negative prompts file, one prompt per line (default: '%s')", params.cvector\_negative\_file.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.cvector\_negative\_file = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_CVECTOR\_GENERATOR}));
 add\_opt(common\_arg(
 {"--pca-batch"}, "N",
 string\_format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n\_pca\_batch),
 \[\](common\_params & params, int value) {
 params.n\_pca\_batch = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_CVECTOR\_GENERATOR}));
 add\_opt(common\_arg(
 {"--pca-iter"}, "N",
 string\_format("number of iterations used for PCA (default: %d)", params.n\_pca\_iterations),
 \[\](common\_params & params, int value) {
 params.n\_pca\_iterations = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_CVECTOR\_GENERATOR}));
 add\_opt(common\_arg(
 {"--method"}, "{pca, mean}",
 "dimensionality reduction method to be used (default: pca)",
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "pca") { params.cvector\_dimre\_method = DIMRE\_METHOD\_PCA; }
 else if (value == "mean") { params.cvector\_dimre\_method = DIMRE\_METHOD\_MEAN; }
 else { throw std::invalid\_argument("invalid value"); }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_CVECTOR\_GENERATOR}));
 add\_opt(common\_arg(
 {"--output-format"}, "{md,jsonl}",
 "output format for batched-bench results (default: md)",
 \[\](common\_params & params, const std::string & value) {
 /\*\*/ if (value == "jsonl") { params.batched\_bench\_output\_jsonl = true; }
 else if (value == "md") { params.batched\_bench\_output\_jsonl = false; }
 else { throw std::invalid\_argument("invalid value"); }
 }
 ).set\_examples({LLAMA\_EXAMPLE\_BENCH}));
 add\_opt(common\_arg(
 {"--log-disable"},
 "Log disable",
 \[\](common\_params &) {
 common\_log\_pause(common\_log\_main());
 }
 ));
 add\_opt(common\_arg(
 {"--log-file"}, "FNAME",
 "Log to file",
 \[\](common\_params &, const std::string & value) {
 common\_log\_set\_file(common\_log\_main(), value.c\_str());
 }
 ).set\_env("LLAMA\_LOG\_FILE"));
 add\_opt(common\_arg(
 {"--log-colors"}, "\[on\|off\|auto\]",
 "Set colored logging ('on', 'off', or 'auto', default: 'auto')\\n"
 "'auto' enables colors when output is to a terminal",
 \[\](common\_params &, const std::string & value) {
 if (is\_truthy(value)) {
 common\_log\_set\_colors(common\_log\_main(), LOG\_COLORS\_ENABLED);
 } else if (is\_falsey(value)) {
 common\_log\_set\_colors(common\_log\_main(), LOG\_COLORS\_DISABLED);
 } else if (is\_autoy(value)) {
 common\_log\_set\_colors(common\_log\_main(), LOG\_COLORS\_AUTO);
 } else {
 throw std::invalid\_argument(
 string\_format("error: unknown value for --log-colors: '%s'\\n", value.c\_str()));
 }
 }
 ).set\_env("LLAMA\_LOG\_COLORS"));
 add\_opt(common\_arg(
 {"-v", "--verbose", "--log-verbose"},
 "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
 \[\](common\_params & params) {
 params.verbosity = INT\_MAX;
 common\_log\_set\_verbosity\_thold(INT\_MAX);
 }
 ));
 add\_opt(common\_arg(
 {"--offline"},
 "Offline mode: forces use of cache, prevents network access",
 \[\](common\_params & params) {
 params.offline = true;
 }
 ).set\_env("LLAMA\_OFFLINE"));
 add\_opt(common\_arg(
 {"-lv", "--verbosity", "--log-verbosity"}, "N",
 string\_format("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\\n"
 " \- 0: generic output\\n"
 " \- 1: error\\n"
 " \- 2: warning\\n"
 " \- 3: info\\n"
 " \- 4: trace (more info)\\n"
 " \- 5: debug\\n"
 "(default: %d)\\n", params.verbosity),
 \[\](common\_params & params, int value) {
 params.verbosity = value;
 common\_log\_set\_verbosity\_thold(value);
 }
 ).set\_env("LLAMA\_LOG\_VERBOSITY"));
 add\_opt(common\_arg(
 {"--log-prefix"},
 {"--no-log-prefix"},
 "Enable prefix in log messages",
 \[\](common\_params &, bool value) {
 common\_log\_set\_prefix(common\_log\_main(), value);
 }
 ).set\_env("LLAMA\_ARG\_LOG\_PREFIX"));
 add\_opt(common\_arg(
 {"--log-timestamps"},
 {"--no-log-timestamps"},
 "Enable timestamps in log messages",
 \[\](common\_params &, bool value) {
 common\_log\_set\_timestamps(common\_log\_main(), value);
 }
 ).set\_env("LLAMA\_ARG\_LOG\_TIMESTAMPS"));

 //
 // speculative parameters
 //

 add\_opt(common\_arg(
 {"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"}, "/\[:quant\]",
 "Same as --hf-repo, but for the draft model (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.mparams.hf\_repo = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_HF\_REPO"));
 add\_opt(common\_arg(
 {"--spec-draft-threads", "-td", "--threads-draft"}, "N",
 "number of threads to use during generation (default: same as --threads)",
 \[\](common\_params & params, int value) {
 params.speculative.draft.cpuparams.n\_threads = value;
 if (params.speculative.draft.cpuparams.n\_threads <= 0) {
 params.speculative.draft.cpuparams.n\_threads = std::thread::hardware\_concurrency();
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-threads-batch", "-tbd", "--threads-batch-draft"}, "N",
 "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
 \[\](common\_params & params, int value) {
 params.speculative.draft.cpuparams\_batch.n\_threads = value;
 if (params.speculative.draft.cpuparams\_batch.n\_threads <= 0) {
 params.speculative.draft.cpuparams\_batch.n\_threads = std::thread::hardware\_concurrency();
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft"}, "M",
 "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
 \[\](common\_params & params, const std::string & mask) {
 params.speculative.draft.cpuparams.mask\_valid = true;
 if (!parse\_cpu\_mask(mask, params.speculative.draft.cpuparams.cpumask)) {
 throw std::invalid\_argument("invalid cpumask");
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-range", "-Crd", "--cpu-range-draft"}, "lo-hi",
 "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
 \[\](common\_params & params, const std::string & range) {
 params.speculative.draft.cpuparams.mask\_valid = true;
 if (!parse\_cpu\_range(range, params.speculative.draft.cpuparams.cpumask)) {
 throw std::invalid\_argument("invalid range");
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-strict", "--cpu-strict-draft"}, "<0\|1>",
 "Use strict CPU placement for draft model (default: same as --cpu-strict)",
 \[\](common\_params & params, int value) {
 params.speculative.draft.cpuparams.strict\_cpu = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-prio", "--prio-draft"}, "N",
 string\_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\\n", params.speculative.draft.cpuparams.priority),
 \[\](common\_params & params, int prio) {
 if (prio < 0 \|\| prio > 3) {
 throw std::invalid\_argument("invalid value");
 }
 params.speculative.draft.cpuparams.priority = (enum ggml\_sched\_priority) prio;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-poll", "--poll-draft"}, "<0\|1>",
 "Use polling to wait for draft model work (default: same as --poll)",
 \[\](common\_params & params, int value) {
 params.speculative.draft.cpuparams.poll = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft"}, "M",
 "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
 \[\](common\_params & params, const std::string & mask) {
 params.speculative.draft.cpuparams\_batch.mask\_valid = true;
 if (!parse\_cpu\_mask(mask, params.speculative.draft.cpuparams\_batch.cpumask)) {
 throw std::invalid\_argument("invalid cpumask");
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
 "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
 \[\](common\_params & params, const std::string & range) {
 params.speculative.draft.cpuparams\_batch.mask\_valid = true;
 if (!parse\_cpu\_range(range, params.speculative.draft.cpuparams\_batch.cpumask)) {
 throw std::invalid\_argument("invalid cpumask");
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft"}, "<0\|1>",
 "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
 \[\](common\_params & params, int value) {
 params.speculative.draft.cpuparams\_batch.strict\_cpu = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-prio-batch", "--prio-batch-draft"}, "N",
 string\_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\\n", params.speculative.draft.cpuparams\_batch.priority),
 \[\](common\_params & params, int prio) {
 if (prio < 0 \|\| prio > 3) {
 throw std::invalid\_argument("invalid value");
 }
 params.speculative.draft.cpuparams\_batch.priority = (enum ggml\_sched\_priority) prio;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-poll-batch", "--poll-batch-draft"}, "<0\|1>",
 "Use polling to wait for draft model work (default: --poll-draft)",
 \[\](common\_params & params, int value) {
 params.speculative.draft.cpuparams\_batch.poll = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"}, "TYPE",
 string\_format(
 "KV cache data type for K for the draft model\\n"
 "allowed values: %s\\n"
 "(default: %s)",
 get\_all\_kv\_cache\_types().c\_str(),
 ggml\_type\_name(params.speculative.draft.cache\_type\_k)
 ),
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.cache\_type\_k = kv\_cache\_type\_from\_str(value);
 }
 ).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_CACHE\_TYPE\_K"));
 add\_opt(common\_arg(
 {"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"}, "TYPE",
 string\_format(
 "KV cache data type for V for the draft model\\n"
 "allowed values: %s\\n"
 "(default: %s)",
 get\_all\_kv\_cache\_types().c\_str(),
 ggml\_type\_name(params.speculative.draft.cache\_type\_v)
 ),
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.cache\_type\_v = kv\_cache\_type\_from\_str(value);
 }
 ).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_CACHE\_TYPE\_V"));
 add\_opt(common\_arg(
 {"--spec-draft-override-tensor", "-otd", "--override-tensor-draft"}, "=,...",
 "override tensor buffer type for draft model", \[\](common\_params & params, const std::string & value) {
 parse\_tensor\_buffer\_overrides(value, params.speculative.draft.tensor\_buft\_overrides);
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"},
 "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
 \[\](common\_params & params) {
 params.speculative.draft.tensor\_buft\_overrides.push\_back(llm\_ffn\_exps\_cpu\_override());
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_CPU\_MOE"));
 add\_opt(common\_arg(
 {"--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"}, "N",
 "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model",
 \[\](common\_params & params, int value) {
 if (value < 0) {
 throw std::invalid\_argument("invalid value");
 }
 for (int i = 0; i < value; ++i) {
 static std::list buft\_overrides\_draft;
 buft\_overrides\_draft.push\_back(llm\_ffn\_exps\_block\_regex(i));
 params.speculative.draft.tensor\_buft\_overrides.push\_back({buft\_overrides\_draft.back().c\_str(), ggml\_backend\_cpu\_buffer\_type()});
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_N\_CPU\_MOE"));

 add\_opt(common\_arg(
 {"--spec-draft-n-max"}, "N",
 string\_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.draft.n\_max),
 \[\](common\_params & params, int value) {
 params.speculative.draft.n\_max = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_LOOKUP, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_N\_MAX"));
 add\_opt(common\_arg(
 {"--spec-draft-n-min"}, "N",
 string\_format("minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.draft.n\_min),
 \[\](common\_params & params, int value) {
 params.speculative.draft.n\_min = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_LOOKUP, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_N\_MIN"));

 add\_opt(common\_arg(
 {"--spec-draft-p-split", "--draft-p-split"}, "P",
 string\_format("speculative decoding split probability (default: %.2f)", (double)params.speculative.draft.p\_split),
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.p\_split = std::stof(value);
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_P\_SPLIT"));
 add\_opt(common\_arg(
 {"--spec-draft-p-min", "--draft-p-min"}, "P",
 string\_format("minimum speculative decoding probability (greedy) (default: %.2f)", (double)params.speculative.draft.p\_min),
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.p\_min = std::stof(value);
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_P\_MIN"));
 add\_opt(common\_arg(
 {"--spec-draft-backend-sampling"},
 {"--no-spec-draft-backend-sampling"},
 string\_format("offload draft sampling to the backend (default: %s)",
 params.speculative.draft.backend\_sampling ? "enabled" : "disabled"),
 \[\](common\_params & params, bool value) {
 params.speculative.draft.backend\_sampling = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_BACKEND\_SAMPLING"));
 add\_opt(common\_arg(
 {"--spec-draft-device", "-devd", "--device-draft"}, "",
 "comma-separated list of devices to use for offloading the draft model (none = don't offload)\\n"
 "use --list-devices to see a list of available devices",
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.devices = parse\_device\_list(value);
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 GGML\_ASSERT(params.speculative.draft.n\_gpu\_layers < 0); // string\_format would need to be extended for a default >= 0
 add\_opt(common\_arg(
 {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
 string\_format("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
 params.speculative.draft.n\_gpu\_layers == -1 ? "auto" : "all"),
 \[\](common\_params & params, const std::string & value) {
 if (value == "auto") {
 params.speculative.draft.n\_gpu\_layers = -1;
 } else if (value == "all") {
 params.speculative.draft.n\_gpu\_layers = -2;
 } else {
 params.speculative.draft.n\_gpu\_layers = std::stoi(value);
 }
 if (!llama\_supports\_gpu\_offload()) {
 fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\\n");
 fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\\n");
 fprintf(stderr, "warning: consult docs/build.md for compilation instructions\\n");
 }
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_N\_GPU\_LAYERS\_DRAFT"));
 add\_opt(common\_arg(
 {"--spec-draft-model", "-md", "--model-draft"}, "FNAME",
 "draft model for speculative decoding (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.speculative.draft.mparams.path = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_DRAFT\_MODEL"));
 add\_opt(common\_arg(
 {"--spec-type"}, common\_speculative\_all\_types\_str(),
 string\_format("comma-separated list of types of speculative decoding to use (default: %s)\\n",
 common\_speculative\_type\_name\_str(params.speculative.types).c\_str()),
 \[\](common\_params & params, const std::string & value) {
 const auto types\_str = string\_split(value, ',');
 auto types = common\_speculative\_types\_from\_names(types\_str);
 params.speculative.types.insert(params.speculative.types.end(), types.begin(), types.end());
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_SPEC\_TYPE"));
 add\_opt(common\_arg(
 {"--spec-ngram-mod-n-min"}, "N",
 string\_format("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram\_mod.n\_min),
 \[\](common\_params & params, int value) {
 if (value < 0 \|\| value > 1024) {
 throw std::invalid\_argument("ngram n-min must be between 0 and 1024 inclusive");
 }
 params.speculative.ngram\_mod.n\_min = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-mod-n-max"}, "N",
 string\_format("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram\_mod.n\_max),
 \[\](common\_params & params, int value) {
 if (value < 0 \|\| value > 1024) {
 throw std::invalid\_argument("ngram n-max must be between 0 and 1024 inclusive");
 }
 params.speculative.ngram\_mod.n\_max = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-mod-n-match"}, "N",
 string\_format("ngram-mod lookup length (default: %d)", params.speculative.ngram\_mod.n\_match),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size N must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_mod.n\_match = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--spec-ngram-simple-size-n"}, "N",
 string\_format("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram\_simple.size\_n),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size N must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_simple.size\_n = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-simple-size-m"}, "N",
 string\_format("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram\_simple.size\_m),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size M must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_simple.size\_m = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-simple-min-hits"}, "N",
 string\_format("minimum hits for ngram-simple speculative decoding (default: %d)", params.speculative.ngram\_simple.min\_hits),
 \[\](common\_params & params, int value) {
 if (value < 1) {
 throw std::invalid\_argument("ngram min hits must be at least 1");
 }
 params.speculative.ngram\_simple.min\_hits = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--spec-ngram-map-k-size-n"}, "N",
 string\_format("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram\_map\_k.size\_n),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size N must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_map\_k.size\_n = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-map-k-size-m"}, "N",
 string\_format("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram\_map\_k.size\_m),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size M must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_map\_k.size\_m = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-map-k-min-hits"}, "N",
 string\_format("minimum hits for ngram-map-k speculative decoding (default: %d)", params.speculative.ngram\_map\_k.min\_hits),
 \[\](common\_params & params, int value) {
 if (value < 1) {
 throw std::invalid\_argument("ngram min hits must be at least 1");
 }
 params.speculative.ngram\_map\_k.min\_hits = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--spec-ngram-map-k4v-size-n"}, "N",
 string\_format("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram\_map\_k4v.size\_n),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size N must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_map\_k4v.size\_n = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-map-k4v-size-m"}, "N",
 string\_format("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram\_map\_k4v.size\_m),
 \[\](common\_params & params, int value) {
 if (value < 1 \|\| value > 1024) {
 throw std::invalid\_argument("ngram size M must be between 1 and 1024 inclusive");
 }
 params.speculative.ngram\_map\_k4v.size\_m = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));
 add\_opt(common\_arg(
 {"--spec-ngram-map-k4v-min-hits"}, "N",
 string\_format("minimum hits for ngram-map-k4v speculative decoding (default: %d)", params.speculative.ngram\_map\_k4v.min\_hits),
 \[\](common\_params & params, int value) {
 if (value < 1) {
 throw std::invalid\_argument("ngram min hits must be at least 1");
 }
 params.speculative.ngram\_map\_k4v.min\_hits = value;
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 //
 // removed params
 //

 add\_opt(common\_arg(
 {"--draft", "--draft-n", "--draft-max"}, "N",
 "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
 \[\](common\_params & /\*params\*/, int /\*value\*/) {
 arg\_removed("use --spec-draft-n-max or --spec-ngram-mod-n-max");
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_LOOKUP, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_DRAFT\_MAX"));
 add\_opt(common\_arg(
 {"--draft-min", "--draft-n-min"}, "N",
 "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
 \[\](common\_params & /\*params\*/, int /\*value\*/) {
 arg\_removed("use --spec-draft-n-min or --spec-ngram-mod-n-min");
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SPECULATIVE, LLAMA\_EXAMPLE\_LOOKUP, LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}).set\_env("LLAMA\_ARG\_DRAFT\_MIN"));
 add\_opt(common\_arg(
 {"--spec-ngram-size-n"}, "N",
 "the argument has been removed. use the respective --spec-ngram-\*-size-n or --spec-ngram-mod-n-match",
 \[\](common\_params & /\*params\*/, int /\*value\*/) {
 arg\_removed("use the respective --spec-ngram-\*-size-n");
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--spec-ngram-size-m"}, "N",
 "the argument has been removed. use the respective --spec-ngram-\*-size-m",
 \[\](common\_params & /\*params\*/, int /\*value\*/) {
 arg\_removed("use the respective --spec-ngram-\*-size-m");
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--spec-ngram-min-hits"}, "N",
 "the argument has been removed. use the respective --spec-ngram-\*-min-hits",
 \[\](common\_params & /\*params\*/, int /\*value\*/) {
 arg\_removed("use the respective --spec-ngram-\*-min-hits");
 }
 ).set\_spec().set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 //
 // TTS params
 //

 add\_opt(common\_arg(
 {"-mv", "--model-vocoder"}, "FNAME",
 "vocoder model for audio generation (default: unused)",
 \[\](common\_params & params, const std::string & value) {
 params.vocoder.model.path = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_TTS, LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--tts-use-guide-tokens"},
 "Use guide tokens to improve TTS word recall",
 \[\](common\_params & params) {
 params.vocoder.use\_guide\_tokens = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_TTS, LLAMA\_EXAMPLE\_SERVER}));
 add\_opt(common\_arg(
 {"--tts-speaker-file"}, "FNAME",
 "speaker file path for audio generation",
 \[\](common\_params & params, const std::string & value) {
 params.vocoder.speaker\_file = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_TTS}));

 //
 // diffusion params
 //

 add\_opt(common\_arg(
 {"--diffusion-steps"}, "N",
 string\_format("number of diffusion steps (default: %d)", params.diffusion.steps),
 \[\](common\_params & params, int value) { params.diffusion.steps = value; }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-visual"},
 string\_format("enable visual diffusion mode (show progressive generation) (default: %s)", params.diffusion.visual\_mode ? "true" : "false"),
 \[\](common\_params & params) { params.diffusion.visual\_mode = true; }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-eps"}, "F",
 string\_format("epsilon for timesteps (default: %.6f)", (double) params.diffusion.eps),
 \[\](common\_params & params, const std::string & value) { params.diffusion.eps = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-algorithm"}, "N",
 string\_format(
 "diffusion algorithm: 0=DIFFUSION\_ALGORITHM\_ORIGIN, 1=DIFFUSION\_ALGORITHM\_ENTROPY\_BASED, "
 "2=DIFFUSION\_ALGORITHM\_MARGIN\_BASED, 3=DIFFUSION\_ALGORITHM\_RANDOM, "
 "4=DIFFUSION\_ALGORITHM\_CONFIDENCE\_BASED (default: %d)", params.diffusion.algorithm),
 \[\](common\_params & params, int value) { params.diffusion.algorithm = value; }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-alg-temp"}, "F",
 string\_format("dream algorithm temperature (default: %.3f)", (double) params.diffusion.alg\_temp),
 \[\](common\_params & params, const std::string & value) { params.diffusion.alg\_temp = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-block-length"}, "N",
 string\_format("llada block length for generation (default: %d)", params.diffusion.block\_length),
 \[\](common\_params & params, int value) { params.diffusion.block\_length = value; }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-cfg-scale"}, "F",
 string\_format("llada classifier-free guidance scale (default: %.3f)", (double) params.diffusion.cfg\_scale),
 \[\](common\_params & params, const std::string & value) { params.diffusion.cfg\_scale = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 {"--diffusion-add-gumbel-noise"}, "F",
 string\_format("add gumbel noise to the logits if temp > 0.0 (default: %s)", params.diffusion.add\_gumbel\_noise ? "true" : "false"),
 \[\](common\_params & params, const std::string & value) { params.diffusion.add\_gumbel\_noise = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_DIFFUSION }));
 add\_opt(common\_arg(
 { "-lr", "--learning-rate" }, "ALPHA",
 string\_format("adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)", (double) params.lr.lr0),
 \[\](common\_params & params, const std::string & value) { params.lr.lr0 = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg({ "-lr-min", "--learning-rate-min" }, "ALPHA",
 string\_format("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
 (double) params.lr.lr\_min),
 \[\](common\_params & params, const std::string & value) { params.lr.lr\_min = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg(
 {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
 string\_format("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)", (double) params.lr.decay\_epochs),
 \[\](common\_params & params, const std::string & value) { params.lr.decay\_epochs = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg(
 {"-wd", "--weight-decay"}, "WD",
 string\_format("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).", (double) params.lr.wd),
 \[\](common\_params & params, const std::string & value) { params.lr.wd = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg(
 {"-val-split", "--val-split"}, "FRACTION",
 string\_format("fraction of data to use as validation set for training (default: %.2g).", (double) params.val\_split),
 \[\](common\_params & params, const std::string & value) { params.val\_split = std::stof(value); }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg(
 {"-epochs", "--epochs"}, "N",
 string\_format("optimizer max # of epochs (default: %d)", params.lr.epochs),
 \[\](common\_params & params, int epochs) { params.lr.epochs = epochs; }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg(
 {"-opt", "--optimizer"}, "sgd\|adamw", "adamw or sgd",
 \[\](common\_params & params, const std::string & name) {
 params.optimizer = common\_opt\_get\_optimizer(name.c\_str());
 if (params.optimizer == GGML\_OPT\_OPTIMIZER\_TYPE\_COUNT) {
 throw std::invalid\_argument("invalid --optimizer, valid options: adamw, sgd");
 }
 }
 ).set\_examples({ LLAMA\_EXAMPLE\_FINETUNE }));
 add\_opt(common\_arg(
 {"--check"},
 string\_format("check rather than generate results (default: %s)", params.check ? "true" : "false"),
 \[\](common\_params & params) {
 params.check = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_RESULTS}));
 add\_opt(common\_arg(
 {"--save-logits"},
 string\_format("save final logits to files for verification (default: %s)", params.save\_logits ? "true" : "false"),
 \[\](common\_params & params) {
 params.save\_logits = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_DEBUG}));
 add\_opt(common\_arg(
 {"--logits-output-dir"}, "PATH",
 string\_format("directory for saving logits output files (default: %s)", params.logits\_output\_dir.c\_str()),
 \[\](common\_params & params, const std::string & value) {
 params.logits\_output\_dir = value;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_DEBUG}));
 add\_opt(common\_arg(
 {"--tensor-filter"}, "REGEX",
 "filter tensor names for debug output (regex pattern, can be specified multiple times)",
 \[\](common\_params & params, const std::string & value) {
 params.tensor\_filter.push\_back(value);
 }
 ).set\_examples({LLAMA\_EXAMPLE\_DEBUG}));

 // presets
 add\_opt(common\_arg(
 {"--tts-oute-default"},
 string\_format("use default OuteTTS models (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "OuteAI/OuteTTS-0.2-500M-GGUF";
 params.model.hf\_file = "OuteTTS-0.2-500M-Q8\_0.gguf";
 params.vocoder.model.hf\_repo = "ggml-org/WavTokenizer";
 params.vocoder.model.hf\_file = "WavTokenizer-Large-75-F16.gguf";
 }
 ).set\_examples({LLAMA\_EXAMPLE\_TTS}));

 add\_opt(common\_arg(
 {"--embd-gemma-default"},
 string\_format("use default EmbeddingGemma model (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/embeddinggemma-300M-qat-q4\_0-GGUF";
 params.model.hf\_file = "embeddinggemma-300M-qat-Q4\_0.gguf";
 params.port = 8011;
 params.n\_ubatch = 2048;
 params.n\_batch = 2048;
 params.n\_parallel = 32;
 params.n\_ctx = 2048\*params.n\_parallel;
 params.verbose\_prompt = true;
 params.embedding = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_EMBEDDING, LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--fim-qwen-1.5b-default"},
 string\_format("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8\_0-GGUF";
 params.model.hf\_file = "qwen2.5-coder-1.5b-q8\_0.gguf";
 params.port = 8012;
 params.n\_ubatch = 1024;
 params.n\_batch = 1024;
 params.n\_ctx = 0;
 params.n\_cache\_reuse = 256;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--fim-qwen-3b-default"},
 string\_format("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/Qwen2.5-Coder-3B-Q8\_0-GGUF";
 params.model.hf\_file = "qwen2.5-coder-3b-q8\_0.gguf";
 params.port = 8012;
 params.n\_ubatch = 1024;
 params.n\_batch = 1024;
 params.n\_ctx = 0;
 params.n\_cache\_reuse = 256;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--fim-qwen-7b-default"},
 string\_format("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/Qwen2.5-Coder-7B-Q8\_0-GGUF";
 params.model.hf\_file = "qwen2.5-coder-7b-q8\_0.gguf";
 params.port = 8012;
 params.n\_ubatch = 1024;
 params.n\_batch = 1024;
 params.n\_ctx = 0;
 params.n\_cache\_reuse = 256;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--fim-qwen-7b-spec"},
 string\_format("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/Qwen2.5-Coder-7B-Q8\_0-GGUF";
 params.model.hf\_file = "qwen2.5-coder-7b-q8\_0.gguf";
 params.speculative.draft.mparams.hf\_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8\_0-GGUF";
 params.speculative.draft.mparams.hf\_file = "qwen2.5-coder-0.5b-q8\_0.gguf";
 params.port = 8012;
 params.n\_ubatch = 1024;
 params.n\_batch = 1024;
 params.n\_ctx = 0;
 params.n\_cache\_reuse = 256;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--fim-qwen-14b-spec"},
 string\_format("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/Qwen2.5-Coder-14B-Q8\_0-GGUF";
 params.model.hf\_file = "qwen2.5-coder-14b-q8\_0.gguf";
 params.speculative.draft.mparams.hf\_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8\_0-GGUF";
 params.speculative.draft.mparams.hf\_file = "qwen2.5-coder-0.5b-q8\_0.gguf";
 params.port = 8012;
 params.n\_ubatch = 1024;
 params.n\_batch = 1024;
 params.n\_ctx = 0;
 params.n\_cache\_reuse = 256;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--fim-qwen-30b-default"},
 string\_format("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8\_0-GGUF";
 params.model.hf\_file = "qwen3-coder-30b-a3b-instruct-q8\_0.gguf";
 params.port = 8012;
 params.n\_ubatch = 1024;
 params.n\_batch = 1024;
 params.n\_ctx = 0;
 params.n\_cache\_reuse = 256;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER}));

 add\_opt(common\_arg(
 {"--gpt-oss-20b-default"},
 string\_format("use gpt-oss-20b (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/gpt-oss-20b-GGUF";
 params.model.hf\_file = "gpt-oss-20b-mxfp4.gguf";
 params.port = 8013;
 params.n\_ubatch = 2048;
 params.n\_batch = 32768;
 params.n\_parallel = 2;
 params.n\_ctx = 131072\*params.n\_parallel;
 params.sampling.temp = 1.0f;
 params.sampling.top\_p = 1.0f;
 params.sampling.top\_k = 0;
 params.sampling.min\_p = 0.01f;
 params.use\_jinja = true;
 //params.default\_template\_kwargs\["reasoning\_effort"\] = "\\"high\\"";
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--gpt-oss-120b-default"},
 string\_format("use gpt-oss-120b (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/gpt-oss-120b-GGUF";
 params.port = 8013;
 params.n\_ubatch = 2048;
 params.n\_batch = 32768;
 params.n\_parallel = 2;
 params.n\_ctx = 131072\*params.n\_parallel;
 params.sampling.temp = 1.0f;
 params.sampling.top\_p = 1.0f;
 params.sampling.top\_k = 0;
 params.sampling.min\_p = 0.01f;
 params.use\_jinja = true;
 //params.default\_template\_kwargs\["reasoning\_effort"\] = "\\"high\\"";
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--vision-gemma-4b-default"},
 string\_format("use Gemma 3 4B QAT (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
 params.port = 8014;
 params.n\_ctx = 0;
 params.use\_jinja = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--vision-gemma-12b-default"},
 string\_format("use Gemma 3 12B QAT (note: can download weights from the internet)"),
 \[\](common\_params & params) {
 params.model.hf\_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
 params.port = 8014;
 params.n\_ctx = 0;
 params.use\_jinja = true;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 add\_opt(common\_arg(
 {"--spec-default"},
 string\_format("enable default speculative decoding config"),
 \[\](common\_params & params) {
 params.speculative.types.push\_back(COMMON\_SPECULATIVE\_TYPE\_NGRAM\_MOD);
 params.speculative.ngram\_mod.n\_match = 24;
 params.speculative.ngram\_mod.n\_min = 48;
 params.speculative.ngram\_mod.n\_max = 64;

 // TODO: not sure if this is a good config - explore more settings and potentially enable it
 //params.speculative.types.push\_back(COMMON\_SPECULATIVE\_TYPE\_NGRAM\_MAP\_K4V);
 //params.speculative.ngram\_map\_k4v.size\_n = 8;
 //params.speculative.ngram\_map\_k4v.size\_m = 24;
 //params.speculative.ngram\_map\_k4v.min\_hits = 2;
 }
 ).set\_examples({LLAMA\_EXAMPLE\_SERVER, LLAMA\_EXAMPLE\_CLI}));

 return ctx\_arg;
}

void common\_params\_add\_preset\_options(std::vector & args) {
 // arguments below won't be treated as CLI args, only preset options
 args.push\_back(common\_arg(
 {"load-on-startup"}, "NAME",
 "in server router mode, autoload this model on startup",
 \[\](common\_params &, const std::string &) { /\* unused \*/ }
 ).set\_env(COMMON\_ARG\_PRESET\_LOAD\_ON\_STARTUP).set\_preset\_only());

 args.push\_back(common\_arg(
 {"stop-timeout"}, "SECONDS",
 "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
 \[\](common\_params &, int) { /\* unused \*/ }
 ).set\_env(COMMON\_ARG\_PRESET\_STOP\_TIMEOUT).set\_preset\_only());

 // args.push\_back(common\_arg(
 // {"pin"},
 // "in server router mode, do not unload this model if models\_max is exceeded",
 // \[\](common\_params &) { /\* unused \*/ }
 // ).set\_preset\_only());
}