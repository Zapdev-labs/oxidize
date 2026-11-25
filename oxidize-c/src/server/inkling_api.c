#define _POSIX_C_SOURCE 200809L
/*
 * inkling_api.c — OpenAI-compatible chat API for Inkling-2: the chat
 * template, the reply parser and the HTTP server (see inkling_api.h).
 */
#include "oxidize/inkling_api.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "oxidize/arena.h"
#include "oxidize/http.h"
#include "oxidize/json.h"
#include "oxidize/log.h"
#include "oxidize/tokenizer.h"

/* ─── text buffer and JSON output ───────────────────────────────────────── */

typedef struct {
    char  *p;
    size_t n, cap;
    bool   oom;
} Buf;

static void b_put(Buf *b, const char *s, size_t n)
{
    if (b->oom) return;
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + n + 1) c *= 2;
        char *q = realloc(b->p, c);
        if (!q) { b->oom = true; return; }
        b->p = q;
        b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}

static void b_str(Buf *b, const char *s) { b_put(b, s, strlen(s)); }

static void b_fmt(Buf *b, const char *fmt, ...)
{
    char t[256];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(t, sizeof t, fmt, ap);
    va_end(ap);
    if (n > 0) b_put(b, t, (size_t)n < sizeof t ? (size_t)n : sizeof t - 1);
}

/* A JSON string as Python's json.dumps(ensure_ascii=False) writes it. */
static void b_jstr(Buf *b, const char *s, size_t n)
{
    b_put(b, "\"", 1);
    size_t run = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        const char *esc = NULL;
        char u[8];
        switch (c) {
        case '"':  esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        default:
            if (c < 0x20) { snprintf(u, sizeof u, "\\u%04x", c); esc = u; }
        }
        if (!esc) { run++; continue; }
        b_put(b, s + i - run, run);
        run = 0;
        b_str(b, esc);
    }
    b_put(b, s + n - run, run);
    b_put(b, "\"", 1);
}

/* Shortest decimal that reads back as v (integral values without a
 * fraction). */
static void b_num(Buf *b, double v)
{
    if (v == floor(v) && fabs(v) < 1e15) { b_fmt(b, "%.0f", v); return; }
    char t[40];
    for (int p = 1; p <= 17; p++) {
        snprintf(t, sizeof t, "%.*g", p, v);
        if (strtod(t, NULL) == v) break;
    }
    b_str(b, t);
}

static int key_cmp(const void *a, const void *b)
{
    return strcmp((*(const OcJson *const *)a)->key, (*(const OcJson *const *)b)->key);
}

/* Compact JSON (separators "," and ":"), object keys sorted when sort (at
 * every level, as json.dumps(sort_keys=True)). */
static void b_json(Buf *b, const OcJson *v, bool sort)
{
    switch (v->type) {
    case OC_JSON_NULL:   b_str(b, "null"); return;
    case OC_JSON_BOOL:   b_str(b, v->boolean ? "true" : "false"); return;
    case OC_JSON_NUMBER: b_num(b, v->number); return;
    case OC_JSON_STRING: b_jstr(b, v->string, strlen(v->string)); return;
    case OC_JSON_ARRAY:
        b_put(b, "[", 1);
        for (const OcJson *c = v->child; c; c = c->next) {
            if (c != v->child) b_put(b, ",", 1);
            b_json(b, c, sort);
        }
        b_put(b, "]", 1);
        return;
    case OC_JSON_OBJECT: {
        size_t n = 0;
        for (const OcJson *c = v->child; c; c = c->next) n++;
        const OcJson **kids = n ? malloc(n * sizeof *kids) : NULL;
        if (n && !kids) { b->oom = true; return; }
        n = 0;
        for (const OcJson *c = v->child; c; c = c->next) kids[n++] = c;
        if (sort && n > 1) qsort(kids, n, sizeof *kids, key_cmp);
        b_put(b, "{", 1);
        for (size_t i = 0; i < n; i++) {
            if (i) b_put(b, ",", 1);
            b_jstr(b, kids[i]->key, strlen(kids[i]->key));
            b_put(b, ":", 1);
            b_json(b, kids[i], sort);
        }
        b_put(b, "}", 1);
        free(kids);
        return;
    }
    }
}

/* ─── chat template ─────────────────────────────────────────────────────── */

static const OcJson *jget(const OcJson *o, const char *key)
{
    return o && o->type == OC_JSON_OBJECT ? oc_json_get(o, key) : NULL;
}

static bool is_str(const OcJson *v) { return v && v->type == OC_JSON_STRING; }

/* Jinja truthiness of a JSON value (absent is false). */
static bool truthy(const OcJson *v)
{
    if (!v) return false;
    switch (v->type) {
    case OC_JSON_NULL:   return false;
    case OC_JSON_BOOL:   return v->boolean;
    case OC_JSON_NUMBER: return v->number != 0.0;
    case OC_JSON_STRING: return v->string[0] != '\0';
    default:             return v->child != NULL;
    }
}

static bool str_is(const OcJson *v, const char *s) { return is_str(v) && strcmp(v->string, s) == 0; }

static const char *role_token(const char *role)
{
    if (!strcmp(role, "user")) return "<|message_user|>";
    if (!strcmp(role, "assistant")) return "<|message_model|>";
    if (!strcmp(role, "system")) return "<|message_system|>";
    if (!strcmp(role, "tool")) return "<|message_tool|>";
    return NULL;
}

/* The template's thinking-effort system message; false with *err set for an
 * unknown or out-of-range effort. */
static bool render_effort(Buf *out, const OcJson *re, float def, const char **err)
{
    static const struct { const char *k; double v; } map[] = {
        { "none", 0.0 }, { "minimal", 0.1 }, { "low", 0.2 }, { "medium", 0.7 },
        { "high", 0.9 }, { "xhigh", 0.99 }, { "max", 0.99 },
    };
    char dt[32];
    snprintf(dt, sizeof dt, "%g", (double)def);
    double num = strtod(dt, NULL);
    if (is_str(re)) {
        const char *s = re->string;
        while (*s == ' ' || *s == '\t' || *s == '\n') s++;
        size_t n = strlen(s);
        while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n')) n--;
        size_t i = 0;
        for (; i < sizeof map / sizeof map[0]; i++)
            if (strlen(map[i].k) == n && !strncmp(map[i].k, s, n)) break;
        if (i == sizeof map / sizeof map[0]) { *err = "unknown reasoning_effort"; return false; }
        num = map[i].v;
    } else if (re && re->type == OC_JSON_NUMBER) {
        num = re->number;
    }
    if (num < 0.0 || num > 0.99) { *err = "reasoning_effort must be in [0.0, 0.99]"; return false; }
    b_str(out, "<|message_system|><|content_text|>Thinking effort level: ");
    if (num == 0.0) b_str(out, "0");
    else b_num(out, num);
    b_str(out, "<|end_message|>");
    return true;
}

/* One text message: header, content tag, text, end. */
static void render_text(Buf *out, const Buf *header, const char *tag, const char *text)
{
    b_put(out, header->p, header->n);
    b_str(out, tag);
    b_str(out, text);
    b_str(out, "<|end_message|>");
}

static bool render_tools(Buf *out, const OcJson *tools, const char **err)
{
    b_str(out, "<|message_system|>tool_declare<|content_xml|>[");
    for (const OcJson *t = tools->child; t; t = t->next) {
        const OcJson *fn = jget(t, "function") ? jget(t, "function") : t;
        const OcJson *name = jget(fn, "name"), *desc = jget(fn, "description"), *par = jget(fn, "parameters");
        const OcJson *type = jget(t, "type");
        if (!name) { *err = "tool without a name"; return false; }
        if (t != tools->child) b_put(out, ",", 1);
        b_str(out, "{\"description\":");
        if (truthy(desc)) b_json(out, desc, true);
        else b_str(out, "\"\"");
        b_str(out, ",\"name\":");
        b_json(out, name, true);
        b_str(out, ",\"parameters\":");
        if (truthy(par)) b_json(out, par, true);
        else b_str(out, "{}");
        b_str(out, ",\"type\":");
        if (truthy(type)) b_json(out, type, true);
        else b_str(out, "\"function\"");
        b_put(out, "}", 1);
    }
    b_str(out, "]<|end_message|>");
    return true;
}

static bool render_tool_call(Buf *out, OcArena *a, const OcJson *msg, const OcJson *tc, const char **err)
{
    const OcJson *fn = jget(tc, "function"), *name = jget(fn, "name");
    if (!is_str(name)) { *err = "tool call function name must be a string"; return false; }
    const OcJson *mname = jget(msg, "name");
    if (truthy(mname) && !str_is(mname, name->string)) { *err = "assistant name does not match its tool call"; return false; }
    const OcJson *args = jget(fn, "arguments");
    if (truthy(args) && args->type == OC_JSON_STRING) {
        OcJson *parsed = NULL;
        if (oc_json_parse(args->string, strlen(args->string), a, &parsed) != OC_OK) {
            *err = "tool call arguments are not valid JSON";
            return false;
        }
        args = parsed;
    }
    if (truthy(args) && args->type != OC_JSON_OBJECT) { *err = "tool call arguments must be an object"; return false; }
    b_str(out, "<|message_model|>");
    b_str(out, name->string);
    b_str(out, "<|content_invoke_tool_json|>{\"name\":");
    b_jstr(out, name->string, strlen(name->string));
    b_str(out, ",\"args\":{");
    if (truthy(args)) {
        size_t n = 0;
        for (const OcJson *c = args->child; c; c = c->next) n++;
        const OcJson **kids = malloc(n * sizeof *kids);
        if (!kids) { *err = "out of memory"; return false; }
        n = 0;
        for (const OcJson *c = args->child; c; c = c->next) kids[n++] = c;
        qsort(kids, n, sizeof *kids, key_cmp);
        for (size_t i = 0; i < n; i++) {
            if (i) b_put(out, ",", 1);
            b_jstr(out, kids[i]->key, strlen(kids[i]->key));
            b_put(out, ":", 1);
            b_json(out, kids[i], false);
        }
        free(kids);
    }
    b_str(out, "}}<|end_message|>");
    return true;
}

static bool render_message(Buf *out, OcArena *a, const OcJson *msgs, const OcJson *m, const char **err)
{
    const OcJson *role = jget(m, "role");
    const char *rtok = is_str(role) ? role_token(role->string) : NULL;
    if (!rtok) { *err = "unknown message role"; return false; }
    const bool assistant = !strcmp(role->string, "assistant");
    Buf header = { 0 };
    b_str(&header, rtok);
    const OcJson *name = jget(m, "name");
    if (!strcmp(role->string, "tool")) {
        const OcJson *id = jget(m, "tool_call_id");
        if (truthy(name) && is_str(name)) {
            b_str(&header, name->string);
        } else if (truthy(id)) {
            for (const OcJson *p = msgs->child; p; p = p->next) {
                if (!str_is(jget(p, "role"), "assistant") || !truthy(jget(p, "tool_calls"))) continue;
                for (const OcJson *tc = jget(p, "tool_calls")->child; tc; tc = tc->next) {
                    const OcJson *tid = jget(tc, "id"), *tname = jget(jget(tc, "function"), "name");
                    if (is_str(tid) && is_str(id) && !strcmp(tid->string, id->string) && is_str(tname))
                        b_str(&header, tname->string);
                }
            }
        }
    } else if (assistant && truthy(name) && is_str(name)) {
        b_str(&header, name->string);
    }
    if (header.oom) { *err = "out of memory"; return false; }

    const OcJson *rc = jget(m, "reasoning_content"), *content = jget(m, "content"), *calls = jget(m, "tool_calls");
    bool ok = true;
    if (assistant && truthy(rc) && is_str(rc)) render_text(out, &header, "<|content_thinking|>", rc->string);
    if (is_str(content) && (content->string[0] || !(assistant && (truthy(calls) || truthy(rc))))) {
        render_text(out, &header, "<|content_text|>", content->string);
    } else if (truthy(content)) {
        if (content->type != OC_JSON_ARRAY) { *err = "message content must be a string or a list of parts"; ok = false; }
        for (const OcJson *part = ok ? content->child : NULL; part; part = part->next) {
            if (is_str(part)) { render_text(out, &header, "<|content_text|>", part->string); continue; }
            const OcJson *type = jget(part, "type"), *text = jget(part, "text");
            if (!type || str_is(type, "text") || str_is(type, "input_text")) {
                render_text(out, &header, "<|content_text|>", is_str(text) ? text->string : "");
            } else if (str_is(type, "image") || str_is(type, "input_image") || str_is(type, "image_url")) {
                *err = "image content is not supported by this server";
                ok = false;
                break;
            } else {
                *err = "unsupported content part type";
                ok = false;
                break;
            }
        }
    }
    if (ok && assistant && truthy(calls)) {
        if (calls->type != OC_JSON_ARRAY) { *err = "tool_calls must be a list"; ok = false; }
        for (const OcJson *tc = ok ? calls->child : NULL; tc && ok; tc = tc->next) ok = render_tool_call(out, a, m, tc, err);
    }
    if (ok && assistant) b_str(out, "<|content_model_end_sampling|>");
    free(header.p);
    return ok;
}

/* With prefix, only the part before the first non-system message: the
 * tool declarations, the leading system messages and the effort message,
 * which stay the same while a conversation goes on. */
static char *chat_render(const char *request_json, size_t len, float default_effort, bool prefix, const char **err)
{
    static const char *const oom = "out of memory";
    OcArena *a = oc_arena_new(len * 2 + 4096);
    if (!a) { *err = oom; return NULL; }
    OcJson *root = NULL;
    if (oc_json_parse(request_json, len, a, &root) != OC_OK || root->type != OC_JSON_OBJECT) {
        oc_arena_free(a);
        *err = "request is not a JSON object";
        return NULL;
    }
    const OcJson *msgs = jget(root, "messages"), *tools = jget(root, "tools");
    if (!msgs || msgs->type != OC_JSON_ARRAY) {
        oc_arena_free(a);
        *err = "missing 'messages' list";
        return NULL;
    }
    Buf out = { 0 }, effort = { 0 };
    bool ok = render_effort(&effort, jget(root, "reasoning_effort"), default_effort, err);
    if (ok && truthy(tools)) {
        if (tools->type != OC_JSON_ARRAY) { *err = "'tools' must be a list"; ok = false; }
        else ok = render_tools(&out, tools, err);
    }
    bool effort_done = false;
    for (const OcJson *m = ok ? msgs->child : NULL; m && ok; m = m->next) {
        if (!effort_done && !str_is(jget(m, "role"), "system")) {
            b_put(&out, effort.p, effort.n);
            effort_done = true;
            if (prefix) break;
        }
        ok = render_message(&out, a, msgs, m, err);
    }
    if (ok && !effort_done) b_put(&out, effort.p, effort.n);
    free(effort.p);
    oc_arena_free(a);
    if (ok && (out.oom || effort.oom)) { *err = oom; ok = false; }
    if (!ok) { free(out.p); return NULL; }
    if (!out.p) return calloc(1, 1);
    return out.p;
}

char *oc_inkling_chat_render(const char *request_json, size_t len, float default_effort, const char **err)
{
    return chat_render(request_json, len, default_effort, false, err);
}

/* ─── reply parsing ─────────────────────────────────────────────────────── */

enum { RS_IDLE, RS_HEADER, RS_THINK, RS_TEXT, RS_TOOL, RS_SKIP };

void oc_inkling_reply_init(OcInklingReply *r, OcInklingPartCb cb, void *user)
{
    memset(r, 0, sizeof *r);
    r->cb = cb;
    r->user = user;
}

void oc_inkling_reply_free(OcInklingReply *r)
{
    free(r->name);
    free(r->json);
    r->name = r->json = NULL;
}

static void grow_put(char **p, size_t *len, size_t *cap, const char *s, size_t n)
{
    if (*len + n + 1 > *cap) {
        size_t c = *cap ? *cap : 128;
        while (c < *len + n + 1) c *= 2;
        char *q = realloc(*p, c);
        if (!q) return;
        *p = q;
        *cap = c;
    }
    memcpy(*p + *len, s, n);
    *len += n;
    (*p)[*len] = '\0';
}

/* A finished tool message: the arguments are the "args" object of its
 * {"name", "args"} body; a body that does not parse goes out as text. */
static void emit_tool(OcInklingReply *r)
{
    const char *body = r->json ? r->json : "";
    OcArena *a = oc_arena_new(r->json_len * 2 + 1024);
    OcJson *root = NULL;
    const bool parsed = a && oc_json_parse(body, r->json_len, a, &root) == OC_OK && root->type == OC_JSON_OBJECT;
    if (!parsed) {
        r->cb(OC_INK_PART_CONTENT, body, r->json_len, NULL, r->user);
        oc_arena_free(a);
        return;
    }
    const char *name = r->name ? r->name : "";
    size_t nlen = r->name_len;
    while (nlen && (name[0] == ' ' || name[0] == '\n')) { name++; nlen--; }
    while (nlen && (name[nlen - 1] == ' ' || name[nlen - 1] == '\n')) nlen--;
    const OcJson *jn = oc_json_get(root, "name");
    if (nlen == 0 && is_str(jn)) { name = jn->string; nlen = strlen(name); }
    const OcJson *args = oc_json_get(root, "args");
    Buf ab = { 0 };
    if (args && args->type == OC_JSON_OBJECT) b_json(&ab, args, false);
    else b_str(&ab, "{}");
    r->n_tool_calls++;
    r->cb(OC_INK_PART_TOOL_CALL, name, nlen, ab.p ? ab.p : "{}", r->user);
    free(ab.p);
    oc_arena_free(a);
}

static bool tok_is(const char *text, size_t len, const char *tok)
{
    return strlen(tok) == len && !memcmp(text, tok, len);
}

static bool tok_starts(const char *text, size_t len, const char *prefix)
{
    const size_t n = strlen(prefix);
    return len > n && !memcmp(text, prefix, n);
}

void oc_inkling_reply_push(OcInklingReply *r, const char *text, size_t len, bool special)
{
    if (r->done || len == 0) return;
    if (special) {
        if (tok_is(text, len, "<|message_model|>")) {
            r->state = RS_HEADER;
            r->name_len = 0;
            if (r->name) r->name[0] = '\0';
        } else if (tok_is(text, len, "<|content_thinking|>")) {
            r->state = RS_THINK;
        } else if (tok_is(text, len, "<|content_text|>")) {
            r->state = RS_TEXT;
        } else if (tok_is(text, len, "<|content_invoke_tool_json|>")) {
            r->state = RS_TOOL;
            r->json_len = 0;
            if (r->json) r->json[0] = '\0';
        } else if (tok_is(text, len, "<|content_model_end_sampling|>") || tok_starts(text, len, "<|message_")) {
            if (r->state == RS_TOOL) emit_tool(r);
            r->done = true;
        } else if (tok_starts(text, len, "<|content_")) {
            r->state = RS_SKIP;
        } else if (tok_is(text, len, "<|end_message|>")) {
            if (r->state == RS_TOOL) emit_tool(r);
            r->state = RS_IDLE;
        }
        return;
    }
    switch (r->state) {
    case RS_HEADER: grow_put(&r->name, &r->name_len, &r->name_cap, text, len); break;
    case RS_THINK:  r->cb(OC_INK_PART_REASONING, text, len, NULL, r->user); break;
    case RS_IDLE:
    case RS_TEXT:   r->cb(OC_INK_PART_CONTENT, text, len, NULL, r->user); break;
    case RS_TOOL:   grow_put(&r->json, &r->json_len, &r->json_cap, text, len); break;
    default:        break;
    }
}

/* ─── server ────────────────────────────────────────────────────────────── */

typedef struct {
    const OcInklingModel *m;
    OcInklingServeOptions o;
    pthread_mutex_t       lock;
    OcInklingSession     *s;
    OcInklingSpec        *sp;
    /* prompt caching: slot 0 holds the end of the system prefix, slot 1 the
     * end of the previous prompt (see oc_inkling_session_save) */
    uint32_t             *cached[OC_INK_SAVE_SLOTS];  /* tokens up to each saved position */
    size_t                n_cached[OC_INK_SAVE_SLOTS];
    bool                  have[OC_INK_SAVE_SLOTS];
} Server;

/* Save the session in slot, remembering the n tokens it has run. */
static void cache_save(Server *srv, uint32_t slot, const uint32_t *ids, size_t n)
{
    free(srv->cached[slot]);
    srv->cached[slot] = malloc(n * sizeof *ids);
    srv->have[slot] = srv->cached[slot] && oc_inkling_session_save(srv->s, slot) == OC_OK;
    if (srv->have[slot]) {
        memcpy(srv->cached[slot], ids, n * sizeof *ids);
        srv->n_cached[slot] = n;
    }
}

/* True when slot holds a strict prefix of ids. */
static bool cache_hit(const Server *srv, uint32_t slot, const uint32_t *ids, size_t n)
{
    return srv->have[slot] && srv->n_cached[slot] < n && !memcmp(srv->cached[slot], ids, srv->n_cached[slot] * sizeof *ids);
}

/* One request's generation. */
typedef struct {
    Server              *srv;
    const OcHttpRequest *req;
    bool                 stream, connected, sent_role;
    const uint32_t      *ids;
    size_t               n_ids;
    bool                 first;
    uint32_t             n_out;
    OcStreamingDetokenizer sd;
    OcInklingReply       reply;
    Buf                  content, reasoning, calls;  /* non-streaming */
    char                 id[48];
    long                 created;
} Gen;

static void chunk_head(Buf *b, const Gen *g)
{
    b_fmt(b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", g->id, g->created);
    b_jstr(b, g->srv->o.model_id, strlen(g->srv->o.model_id));
    b_str(b, ",\"choices\":[{\"index\":0,\"delta\":{");
}

static void send_chunk(Gen *g, Buf *b)
{
    if (b->oom || !g->connected) return;
    g->connected = g->req->stream_write(g->req->stream_context, b->p, b->n);
}

static void tool_call_json(Buf *b, uint32_t index, const char *name, size_t nlen, const char *args)
{
    b_fmt(b, "{\"index\":%u,\"id\":\"call_%u\",\"type\":\"function\",\"function\":{\"name\":", index, index);
    b_jstr(b, name, nlen);
    b_str(b, ",\"arguments\":");
    b_jstr(b, args, strlen(args));
    b_str(b, "}}");
}

static void on_part(OcInklingPartKind kind, const char *text, size_t len, const char *args, void *user)
{
    Gen *g = user;
    if (!g->stream) {
        if (kind == OC_INK_PART_REASONING) b_put(&g->reasoning, text, len);
        else if (kind == OC_INK_PART_CONTENT) b_put(&g->content, text, len);
        else {
            if (g->calls.n) b_put(&g->calls, ",", 1);
            tool_call_json(&g->calls, g->reply.n_tool_calls - 1, text, len, args);
        }
        return;
    }
    Buf b = { 0 };
    chunk_head(&b, g);
    if (!g->sent_role) b_str(&b, "\"role\":\"assistant\",");
    g->sent_role = true;
    if (kind == OC_INK_PART_TOOL_CALL) {
        b_str(&b, "\"tool_calls\":[");
        tool_call_json(&b, g->reply.n_tool_calls - 1, text, len, args);
        b_put(&b, "]", 1);
    } else {
        b_str(&b, kind == OC_INK_PART_REASONING ? "\"reasoning_content\":" : "\"content\":");
        b_jstr(&b, text, len);
    }
    b_str(&b, "},\"finish_reason\":null}]}\n\n");
    send_chunk(g, &b);
    free(b.p);
}

/* Inkling's control tokens (<|message_model|>, <|end_message|>, ...) are
 * added tokens of the form <|name|>; the reply parser needs them whole. */
static bool is_control(const char *piece)
{
    const size_t n = strlen(piece);
    if (n < 5 || strncmp(piece, "<|", 2) != 0 || strcmp(piece + n - 2, "|>") != 0) return false;
    for (size_t i = 2; i < n - 2; i++)
        if (!(piece[i] == '_' || (piece[i] >= 'a' && piece[i] <= 'z') || (piece[i] >= '0' && piece[i] <= '9'))) return false;
    return true;
}

static bool on_token(uint32_t tok, void *user)
{
    Gen *g = user;
    Server *srv = g->srv;
    const OcTokenizer *t = &srv->m->tok;
    if (g->first) {
        /* the prompt is in: keep this point for the next request */
        g->first = false;
        cache_save(srv, 1, g->ids, g->n_ids);
    }
    g->n_out++;
    const char *d = NULL;
    size_t dn = 0;
    char *piece = NULL;
    if (oc_tokenizer_decode(t, &tok, 1, &piece) == OC_OK && piece && is_control(piece)) {
        if (oc_streaming_detok_flush(&g->sd, &d, &dn) == OC_OK && dn) oc_inkling_reply_push(&g->reply, d, dn, false);
        oc_inkling_reply_push(&g->reply, piece, strlen(piece), true);
    } else if (oc_streaming_detok_push(&g->sd, tok, &d, &dn) == OC_OK && dn) {
        oc_inkling_reply_push(&g->reply, d, dn, false);
    }
    free(piece);
    return !g->reply.done && (!g->stream || g->connected);
}

static char *error_json(const char *msg, const char *type)
{
    Buf b = { 0 };
    b_str(&b, "{\"error\":{\"message\":");
    b_jstr(&b, msg, strlen(msg));
    b_str(&b, ",\"type\":");
    b_jstr(&b, type, strlen(type));
    b_str(&b, "}}");
    return b.p;
}

static void respond(int *status, const char **ctype, const char **body, size_t *body_len, int code, char *json)
{
    *status = code;
    *ctype = "application/json";
    *body = json;
    *body_len = json ? strlen(json) : 0;
}

static bool request_bool(const OcJson *root, const char *key)
{
    const OcJson *v = jget(root, key);
    return v && v->type == OC_JSON_BOOL && v->boolean;
}

static void handle_chat(Server *srv, const OcHttpRequest *req, int *status, const char **ctype, const char **body,
                        size_t *body_len)
{
    const char *err = NULL;
    char *prompt = oc_inkling_chat_render(req->body, req->content_length, srv->o.effort, &err);
    if (!prompt) { respond(status, ctype, body, body_len, 400, error_json(err, "invalid_request_error")); return; }
    OcArena *a = oc_arena_new(req->content_length + 1024);
    OcJson *root = NULL;
    if (!a || oc_json_parse(req->body, req->content_length, a, &root) != OC_OK) {
        free(prompt);
        oc_arena_free(a);
        respond(status, ctype, body, body_len, 400, error_json("invalid JSON", "invalid_request_error"));
        return;
    }
    uint32_t max_tokens = srv->o.n_predict;
    const OcJson *mt = jget(root, "max_completion_tokens");
    if (!mt || mt->type != OC_JSON_NUMBER) mt = jget(root, "max_tokens");
    if (mt && mt->type == OC_JSON_NUMBER && mt->number >= 1) max_tokens = (uint32_t)fmin(mt->number, 1e9);
    const bool stream = req->stream_write != NULL;
    const bool usage_chunk = request_bool(jget(root, "stream_options"), "include_usage");
    oc_arena_free(a);

    uint32_t *ids = NULL, *sys_ids = NULL;
    size_t n_ids = 0, n_sys = 0;
    const OcError te = oc_tokenizer_encode(&srv->m->tok, prompt, OC_TOK_ALLOW_SPECIAL, &ids, &n_ids);
    free(prompt);
    char *sys = chat_render(req->body, req->content_length, srv->o.effort, true, &err);
    if (sys && oc_tokenizer_encode(&srv->m->tok, sys, OC_TOK_ALLOW_SPECIAL, &sys_ids, &n_sys) != OC_OK) n_sys = 0;
    free(sys);
    if (te == OC_OK && (n_sys >= n_ids || memcmp(sys_ids, ids, n_sys * sizeof *ids) != 0)) n_sys = 0;
    free(sys_ids);
    if (te != OC_OK || n_ids == 0) {
        free(ids);
        respond(status, ctype, body, body_len, 400, error_json("cannot tokenize the conversation", "invalid_request_error"));
        return;
    }
    if (n_ids >= srv->o.n_ctx) {
        free(ids);
        respond(status, ctype, body, body_len, 400,
                error_json("the conversation is longer than the server's context (--ctx)", "invalid_request_error"));
        return;
    }
    if (max_tokens > srv->o.n_ctx - n_ids) max_tokens = srv->o.n_ctx - (uint32_t)n_ids;

    Gen g = { .srv = srv, .req = req, .stream = stream, .connected = true, .ids = ids, .n_ids = n_ids, .first = true };
    snprintf(g.id, sizeof g.id, "chatcmpl-%lx%04x", (unsigned long)time(NULL), (unsigned)(rand() & 0xffff));
    g.created = (long)time(NULL);
    oc_streaming_detok_init(&g.sd, &srv->m->tok);
    oc_inkling_reply_init(&g.reply, on_part, &g);

    pthread_mutex_lock(&srv->lock);
    size_t reuse = 0;
    if (cache_hit(srv, 1, ids, n_ids) && oc_inkling_session_restore(srv->s, 1) == OC_OK) {
        reuse = srv->n_cached[1];
    } else if (cache_hit(srv, 0, ids, n_ids) && oc_inkling_session_restore(srv->s, 0) == OC_OK) {
        reuse = srv->n_cached[0];
        srv->have[1] = false;
    } else {
        oc_inkling_session_reset(srv->s);
        srv->have[0] = srv->have[1] = false;
    }
    const size_t cached = reuse;
    OcError ge = OC_OK;
    if (reuse < n_sys) {
        /* run up to the end of the system prefix and keep that point */
        srv->have[1] = false;
        const size_t chunk = oc_inkling_session_max_batch(srv->s);
        for (size_t i = reuse; i < n_sys && ge == OC_OK; i += chunk) {
            const size_t c = n_sys - i < chunk ? n_sys - i : chunk;
            ge = oc_inkling_forward(srv->s, ids + i, c, NULL, false);
            if (ge == OC_OK && srv->sp) ge = oc_inkling_spec_inject(srv->sp, c);
        }
        if (ge == OC_OK) cache_save(srv, 0, ids, n_sys);
        reuse = n_sys;
    }
    OcInklingSpecStats st = { 0 };
    if (ge == OC_OK)
        ge = oc_inkling_generate(srv->s, srv->sp, ids + reuse, n_ids - reuse, max_tokens,
                                 srv->sp ? srv->o.draft_max : 0, srv->o.p_min, on_token, &g, &st);
    if (ge != OC_OK) srv->have[0] = srv->have[1] = false;
    pthread_mutex_unlock(&srv->lock);
    const char *d = NULL;
    size_t dn = 0;
    if (oc_streaming_detok_flush(&g.sd, &d, &dn) == OC_OK && dn) oc_inkling_reply_push(&g.reply, d, dn, false);
    if (g.reply.state == RS_TOOL && !g.reply.done) {
        g.reply.done = true;
        emit_tool(&g.reply);
    }
    oc_log(OC_LOG_INFO, "inkling-api: %zu prompt tokens (%zu cached), %u generated, prefill %.1f tok/s, decode %.1f tok/s",
           n_ids, cached, g.n_out, st.t_prefill > 0 ? (double)st.n_prompt / st.t_prefill : 0.0,
           st.t_decode > 0 && st.emitted > 1 ? (double)(st.emitted - 1) / st.t_decode : 0.0);

    const char *finish = g.reply.n_tool_calls ? "tool_calls" : (!g.reply.done && g.n_out >= max_tokens ? "length" : "stop");
    Buf usage = { 0 };
    b_fmt(&usage, "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%u,\"total_tokens\":%zu,"
                  "\"prompt_tokens_details\":{\"cached_tokens\":%zu}}", n_ids, g.n_out, n_ids + g.n_out, cached);
    Buf out = { 0 };
    if (ge != OC_OK && !g.n_out) {
        char *ej = error_json(oc_error_msg(ge), "server_error");
        if (stream) {
            b_str(&out, "data: ");
            b_str(&out, ej);
            b_str(&out, "\n\ndata: [DONE]\n\n");
            send_chunk(&g, &out);
            free(ej);
            respond(status, ctype, body, body_len, 200, NULL);
        } else {
            respond(status, ctype, body, body_len, 500, ej);
        }
    } else if (stream) {
        chunk_head(&out, &g);
        b_str(&out, "},\"finish_reason\":");
        b_jstr(&out, finish, strlen(finish));
        b_str(&out, "}]");
        if (usage_chunk) { b_put(&out, ",", 1); b_put(&out, usage.p, usage.n); }
        b_str(&out, "}\n\ndata: [DONE]\n\n");
        send_chunk(&g, &out);
        respond(status, ctype, body, body_len, 200, NULL);
    } else {
        b_fmt(&out, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", g.id, g.created);
        b_jstr(&out, srv->o.model_id, strlen(srv->o.model_id));
        b_str(&out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        if (g.content.n || !g.calls.n) b_jstr(&out, g.content.p ? g.content.p : "", g.content.n);
        else b_str(&out, "null");
        if (g.reasoning.n) {
            b_str(&out, ",\"reasoning_content\":");
            b_jstr(&out, g.reasoning.p, g.reasoning.n);
        }
        if (g.calls.n) {
            b_str(&out, ",\"tool_calls\":[");
            b_put(&out, g.calls.p, g.calls.n);
            b_put(&out, "]", 1);
        }
        b_str(&out, "},\"finish_reason\":");
        b_jstr(&out, finish, strlen(finish));
        b_str(&out, "}],");
        b_put(&out, usage.p, usage.n);
        b_put(&out, "}", 1);
        respond(status, ctype, body, body_len, out.oom ? 500 : 200, out.oom ? NULL : out.p);
        out.p = out.oom ? out.p : NULL;
    }
    free(out.p);
    free(usage.p);
    free(g.content.p);
    free(g.reasoning.p);
    free(g.calls.p);
    oc_inkling_reply_free(&g.reply);
    oc_streaming_detok_free(&g.sd);
    free(ids);
}

static void handler(const OcHttpRequest *req, int *status, const char **ctype, const char **body, size_t *body_len,
                    void *user)
{
    Server *srv = user;
    OcRequestContext rctx = { req->method, req->path, req->auth_header, req->client_ip };
    if (srv->o.mw) {
        const int reject = oc_middleware_process_request(srv->o.mw, &rctx);
        if (reject) {
            respond(status, ctype, body, body_len, reject,
                    error_json(reject == 401 ? "unauthorized" : "rate limit exceeded",
                               reject == 401 ? "authentication_error" : "rate_limit_error"));
            return;
        }
    }
    if (req->method == OC_HTTP_GET && (!strcmp(req->path, "/healthz") || !strcmp(req->path, "/livez") ||
                                       !strcmp(req->path, "/readyz"))) {
        Buf b = { 0 };
        b_str(&b, "{\"status\":\"ok\"}");
        respond(status, ctype, body, body_len, 200, b.p);
    } else if (req->method == OC_HTTP_GET && !strcmp(req->path, "/v1/models")) {
        Buf b = { 0 };
        b_str(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
        b_jstr(&b, srv->o.model_id, strlen(srv->o.model_id));
        b_fmt(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"oxidize\",\"context_length\":%u}]}", srv->o.n_ctx);
        respond(status, ctype, body, body_len, 200, b.p);
    } else if (req->method == OC_HTTP_POST && !strcmp(req->path, "/v1/chat/completions")) {
        handle_chat(srv, req, status, ctype, body, body_len);
    } else {
        respond(status, ctype, body, body_len, 404, error_json("not found", "invalid_request_error"));
    }
    if (srv->o.mw) {
        const OcResponseContext resp = { *status, 0, 0 };
        oc_middleware_process_response(srv->o.mw, &rctx, &resp);
    }
}

/* Requests that would stream are checked before the event stream opens, so
 * a bad request still gets an HTTP error status. */
static bool authorize_stream(const OcHttpRequest *req, int *status, const char **body, void *user)
{
    Server *srv = user;
    if (srv->o.mw) {
        OcRequestContext rctx = { req->method, req->path, req->auth_header, req->client_ip };
        const int reject = oc_middleware_process_request(srv->o.mw, &rctx);
        if (reject) {
            *status = reject;
            *body = error_json(reject == 401 ? "unauthorized" : "rate limit exceeded",
                               reject == 401 ? "authentication_error" : "rate_limit_error");
            return false;
        }
    }
    if (strcmp(req->path, "/v1/chat/completions") != 0) {
        *status = 404;
        *body = error_json("streaming is only served on /v1/chat/completions", "invalid_request_error");
        return false;
    }
    const char *err = NULL;
    char *prompt = oc_inkling_chat_render(req->body, req->content_length, srv->o.effort, &err);
    if (!prompt) {
        *status = 400;
        *body = error_json(err, "invalid_request_error");
        return false;
    }
    free(prompt);
    return true;
}

OcError oc_inkling_serve(const OcInklingModel *m, const OcInklingDraft *d, bool gpu, const OcInklingServeOptions *o)
{
    if (!m || !o || !m->have_tok || o->n_ctx < 16) return OC_ERR_INVALID_ARG;
    Server srv = { .m = m, .o = *o };
    if (!srv.o.model_id) srv.o.model_id = "inkling-2";
    pthread_mutex_init(&srv.lock, NULL);
    OcError e = oc_inkling_session_new(m, o->n_ctx, d ? d->target_layers : NULL, d ? d->n_targets : 0, &srv.s);
    if (e == OC_OK && gpu) e = oc_inkling_session_use_gpu(srv.s);
    if (e == OC_OK && d) e = oc_inkling_spec_new(d, srv.s, o->n_ctx, &srv.sp);
    OcHttpServer http;
    memset(&http, 0, sizeof http);
    if (e == OC_OK) {
        oc_http_server_set_stream_authorizer(&http, authorize_stream);
        if (o->mw) {
            char cors[384];
            if (oc_middleware_cors_headers(o->mw, cors, sizeof cors) > 0) oc_http_server_set_extra_headers(&http, cors);
        }
        e = oc_http_server_start_configured(o->host, o->port, 4, handler, &srv, &http);
    }
    if (e == OC_OK) {
        printf("oxidize-c Inkling-2 API on http://%s:%u (model \"%s\", %s%s, %u-token context)\n"
               "  GET  /v1/models\n  POST /v1/chat/completions\n(Ctrl+C to stop)\n",
               o->host, http.port, srv.o.model_id, gpu ? "GPU" : "CPU", d ? " + DSpark" : "", o->n_ctx);
        fflush(stdout);
        oc_http_server_join(&http);
    }
    oc_inkling_spec_free(srv.sp);
    oc_inkling_session_free(srv.s);
    for (int i = 0; i < OC_INK_SAVE_SLOTS; i++) free(srv.cached[i]);
    pthread_mutex_destroy(&srv.lock);
    return e;
}
