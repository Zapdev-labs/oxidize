#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg)
{
    fprintf(stderr, "qwen36_record: %s\n", msg);
    exit(2);
}

static void json_escape(FILE *out, const char *s)
{
    fputc('"', out);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"':  fputs("\\\"", out); break;
        case '\\': fputs("\\\\", out); break;
        case '\n': fputs("\\n", out); break;
        case '\r': fputs("\\r", out); break;
        case '\t': fputs("\\t", out); break;
        default:
            if (c < 0x20) fprintf(out, "\\u%04x", c);
            else fputc(c, out);
        }
    }
    fputc('"', out);
}

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open file");
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); die("seek"); }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); die("tell"); }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); die("oom"); }
    size_t rd = fread(buf, 1, (size_t)n, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

static const char *find_key(const char *s, const char *key)
{
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = s;
    while ((p = strstr(p, pat)) != NULL) {
        const char *q = p + strlen(pat);
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q == ':') return q + 1;
        p++;
    }
    return NULL;
}

static int parse_double_after(const char *s, const char *key, double *out)
{
    const char *v = find_key(s, key);
    if (!v) return 0;
    while (*v && isspace((unsigned char)*v)) v++;
    char *end = NULL;
    *out = strtod(v, &end);
    return end != v;
}

static int parse_int_after(const char *s, const char *key, long *out)
{
    const char *v = find_key(s, key);
    if (!v) return 0;
    while (*v && isspace((unsigned char)*v)) v++;
    char *end = NULL;
    *out = strtol(v, &end, 10);
    return end != v;
}

static double clock_seconds(const char *value)
{
    double parts[4] = {0};
    int n = 0;
    const char *p = value;
    while (*p && n < 4) {
        char *end = NULL;
        parts[n++] = strtod(p, &end);
        if (end == p) break;
        p = end;
        if (*p == ':') p++;
        else break;
    }
    double total = 0.0;
    for (int i = 0; i < n; i++) total = total * 60.0 + parts[i];
    return total;
}

static double elapsed_from_time_file(const char *path)
{
    char *buf = read_file(path);
    double sec = 0.0;
    char *line = buf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (strstr(line, "Elapsed (wall clock) time")) {
            const char *val = strstr(line, "): ");
            if (val) {
                val += 3;
                while (*val && isspace((unsigned char)*val)) val++;
                sec = clock_seconds(val);
            }
        }
        line = nl ? nl + 1 : NULL;
    }
    free(buf);
    return sec;
}

static int oxidize_rates(const char *path, double *prefill, double *decode)
{
    char *buf = read_file(path);
    int ok = 0;
    char *line = buf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        double p = 0, d = 0;
        if (parse_double_after(line, "prefill_tok_per_s", &p) &&
            parse_double_after(line, "decode_tok_per_s", &d)) {
            *prefill = p;
            *decode = d;
            ok = 1;
        }
        line = nl ? nl + 1 : NULL;
    }
    free(buf);
    return ok;
}

static int llama_rates(const char *path, double *prefill, double *decode)
{
    char *buf = read_file(path);
    int got_p = 0, got_d = 0;
    char *line = buf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        long n_prompt = 0, n_gen = 0;
        double avg = 0;
        if (parse_int_after(line, "n_prompt", &n_prompt) &&
            parse_int_after(line, "n_gen", &n_gen) &&
            parse_double_after(line, "avg_ts", &avg)) {
            if (n_prompt > 0 && n_gen == 0) { *prefill = avg; got_p = 1; }
            if (n_gen > 0) { *decode = avg; got_d = 1; }
        }
        line = nl ? nl + 1 : NULL;
    }
    free(buf);
    return got_p && got_d;
}

static const char *need_arg(int *i, int argc, char **argv)
{
    if (*i + 1 >= argc) die("missing flag value");
    return argv[++(*i)];
}

int main(int argc, char **argv)
{
    if (argc < 2) die("usage: qwen36_record manifest|row ...");

    if (strcmp(argv[1], "manifest") == 0) {
        if (argc != 16) die("manifest needs 14 fields after the verb");
        const char *run_dir = argv[2];
        const char *revision = argv[3];
        const char *oxidize_binary = argv[4];
        const char *oxidize_sha256 = argv[5];
        const char *llama_commit = argv[6];
        const char *model_path = argv[7];
        const char *model_sha256 = argv[8];
        const char *model_size = argv[9];
        const char *affinity = argv[10];
        const char *numa = argv[11];
        const char *threads = argv[12];
        const char *warmup = argv[13];
        const char *repetitions = argv[14];
        const char *label = argv[15];
        fputs("{\"affinity\":", stdout); json_escape(stdout, affinity);
        fputs(",\"label\":", stdout); json_escape(stdout, label);
        fputs(",\"llama_commit\":", stdout); json_escape(stdout, llama_commit);
        fputs(",\"model_path\":", stdout); json_escape(stdout, model_path);
        fputs(",\"model_sha256\":", stdout); json_escape(stdout, model_sha256);
        fprintf(stdout, ",\"model_size_bytes\":%ld", strtol(model_size, NULL, 10));
        fprintf(stdout, ",\"numa_node\":%ld", strtol(numa, NULL, 10));
        fputs(",\"oxidize_binary\":", stdout); json_escape(stdout, oxidize_binary);
        fputs(",\"oxidize_sha256\":", stdout); json_escape(stdout, oxidize_sha256);
        fprintf(stdout, ",\"repetitions\":%ld", strtol(repetitions, NULL, 10));
        fputs(",\"revision\":", stdout); json_escape(stdout, revision);
        fputs(",\"run_dir\":", stdout); json_escape(stdout, run_dir);
        fputs(",\"schema\":\"qwen36-cpu-benchmark-v1\"", stdout);
        fprintf(stdout, ",\"threads\":%ld", strtol(threads, NULL, 10));
        fprintf(stdout, ",\"warmup\":%ld}\n", strtol(warmup, NULL, 10));
        return 0;
    }

    if (strcmp(argv[1], "row") != 0) die("unknown verb");

    const char *engine = NULL, *case_name = NULL, *phase = NULL, *run = NULL;
    const char *command = NULL, *timing = NULL, *stdout_path = NULL;
    const char *load_before = NULL, *load_after = NULL, *rss = NULL;
    const char *model = NULL, *sha = NULL, *size = NULL, *revision = NULL;
    const char *llama = NULL, *cpus = NULL, *numa = NULL, *threads = NULL;
    const char *label = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--engine") == 0) engine = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--case") == 0) case_name = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--phase") == 0) phase = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--run") == 0) run = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--command") == 0) command = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--timing") == 0) timing = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--stdout") == 0) stdout_path = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--load-before") == 0) load_before = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--load-after") == 0) load_after = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--rss") == 0) rss = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--model") == 0) model = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--sha") == 0) sha = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--size") == 0) size = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--revision") == 0) revision = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--llama") == 0) llama = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--affinity") == 0) cpus = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--numa") == 0) numa = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--threads") == 0) threads = need_arg(&i, argc, argv);
        else if (strcmp(argv[i], "--label") == 0) label = need_arg(&i, argc, argv);
        else die("unknown row flag");
    }
    if (!engine || !case_name || !phase || !run || !command || !timing || !stdout_path ||
        !load_before || !load_after || !rss || !model || !sha || !size || !revision ||
        !llama || !cpus || !numa || !threads || !label)
        die("row is missing a required flag");

    double prefill = 0, decode = 0;
    if (strcmp(engine, "oxidize-c") == 0) {
        if (!oxidize_rates(stdout_path, &prefill, &decode)) die("oxidize-c stdout missing rates");
    } else if (!llama_rates(stdout_path, &prefill, &decode)) {
        die("llama stdout missing rates");
    }

    double elapsed = elapsed_from_time_file(timing);

    fputs("{\"affinity\":", stdout); json_escape(stdout, cpus);
    fputs(",\"case\":", stdout); json_escape(stdout, case_name);
    fputs(",\"command\":", stdout); json_escape(stdout, command);
    fprintf(stdout, ",\"decode_tok_per_s\":%.17g", decode);
    fputs(",\"engine\":", stdout); json_escape(stdout, engine);
    fputs(",\"label\":", stdout); json_escape(stdout, label);
    fputs(",\"llama_commit\":", stdout); json_escape(stdout, llama);
    fprintf(stdout, ",\"load_after\":%.17g", strtod(load_after, NULL));
    fprintf(stdout, ",\"load_before\":%.17g", strtod(load_before, NULL));
    fputs(",\"mmap\":true", stdout);
    fputs(",\"model_path\":", stdout); json_escape(stdout, model);
    fputs(",\"model_sha256\":", stdout); json_escape(stdout, sha);
    fprintf(stdout, ",\"model_size_bytes\":%ld", strtol(size, NULL, 10));
    fprintf(stdout, ",\"numa_node\":%ld", strtol(numa, NULL, 10));
    fputs(",\"phase\":", stdout); json_escape(stdout, phase);
    fprintf(stdout, ",\"prefill_tok_per_s\":%.17g", prefill);
    fprintf(stdout, ",\"rss_kb\":%ld", strtol(rss, NULL, 10));
    fprintf(stdout, ",\"run\":%ld", strtol(run, NULL, 10));
    fputs(",\"revision\":", stdout); json_escape(stdout, revision);
    fputs(",\"schema\":\"qwen36-cpu-benchmark-v1\"", stdout);
    fprintf(stdout, ",\"threads\":%ld", strtol(threads, NULL, 10));
    fprintf(stdout, ",\"timing\":{\"elapsed_s\":%.17g}}\n", elapsed);
    return 0;
}
