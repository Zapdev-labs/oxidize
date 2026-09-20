#define _POSIX_C_SOURCE 200809L

#include "oxidize/tui.h"
#include "oxidize/cli_commands.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

enum { VIEW_CHAT, VIEW_MODELS, VIEW_MONITOR, VIEW_LOGS };

typedef struct {
    char name[160];
    char path[512];
    unsigned long size;
} TuiModel;

typedef struct {
    char user[2048];
    char asst[8192];
} TuiTurn;

typedef struct {
    int view;
    int generating;
    char draft[1024];
    TuiTurn turns[32];
    int nturns;
    TuiModel models[256];
    int nmodels;
    int mcursor;
    char logs[64][240];
    int nlogs;
    char status[160];
    char url[160];
    char model_id[128];
    char metrics[240];
    pid_t child;
    int overlay; /* 0 none, 1 palette, 2 help */
    char palq[64];
    int palcur;
    float temperature;
    int max_tokens;
} TuiState;

static struct termios g_orig;
static int g_raw;

static const char *PAL[] = {
    "Go to chat",
    "Go to models",
    "Go to monitor",
    "Go to logs",
    "Clear conversation",
    "Stop server",
};

bool oc_tui_fuzzy(const char *query, const char *hay)
{
    if (!query || query[0] == '\0') return true;
    if (!hay) return false;
    size_t ql = strlen(query);
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        const char *q = p;
        while (query[i] && *q &&
               tolower((unsigned char)query[i]) == tolower((unsigned char)*q)) {
            i++;
            q++;
        }
        if (i == ql) return true;
    }
    return false;
}

int oc_tui_sse_delta(const char *line, char *out, size_t out_cap)
{
    if (!line || !out || out_cap == 0) return 0;
    out[0] = '\0';
    if (strncmp(line, "data:", 5) != 0) return 0;
    const char *p = line + 5;
    while (*p == ' ') p++;
    if (p[0] == '\0' || strcmp(p, "[DONE]") == 0) return 0;
    const char *key = strstr(p, "\"content\":\"");
    if (!key) return 0;
    key += 11;
    size_t n = 0;
    while (*key && *key != '"' && n + 1 < out_cap) {
        if (*key == '\\' && key[1]) {
            key++;
            out[n++] = *key++;
        } else {
            out[n++] = *key++;
        }
    }
    out[n] = '\0';
    return n > 0;
}

double oc_tui_metric_value(const char *text, const char *name)
{
    if (!text || !name) return 0.0;
    double sum = 0.0;
    const char *p = text;
    size_t nl = strlen(name);
    while (p && *p) {
        const char *nlpos = strchr(p, '\n');
        size_t len = nlpos ? (size_t)(nlpos - p) : strlen(p);
        if (len > 0 && p[0] != '#' && strncmp(p, name, nl) == 0 &&
            (p[nl] == ' ' || p[nl] == '{')) {
            const char *sp = p + len;
            while (sp > p && sp[-1] != ' ') sp--;
            sum += strtod(sp, NULL);
        }
        p = nlpos ? nlpos + 1 : NULL;
    }
    return sum;
}

static void raw_on(void)
{
    if (tcgetattr(STDIN_FILENO, &g_orig) != 0) return;
    struct termios t = g_orig;
    t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    g_raw = 1;
    write(STDOUT_FILENO, "\x1b[?1049h\x1b[?25l", 16);
}

static void raw_off(void)
{
    if (!g_raw) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &g_orig);
    write(STDOUT_FILENO, "\x1b[?25h\x1b[?1049l", 16);
    g_raw = 0;
}

static void log_line(TuiState *s, const char *msg)
{
    if (s->nlogs >= 64) {
        memmove(s->logs[0], s->logs[1], sizeof(s->logs) - sizeof(s->logs[0]));
        s->nlogs = 63;
    }
    snprintf(s->logs[s->nlogs++], sizeof(s->logs[0]), "%s", msg);
}

static int http_req(const char *host, int port, const char *req, char *buf, size_t cap)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    (void)host;
    struct timeval tv = { .tv_sec = 2, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) {
        close(fd);
        return -1;
    }
    if (write(fd, req, strlen(req)) < 0) {
        close(fd);
        return -1;
    }
    size_t n = 0;
    while (n + 1 < cap) {
        ssize_t r = read(fd, buf + n, cap - 1 - n);
        if (r <= 0) break;
        n += (size_t)r;
    }
    buf[n] = '\0';
    close(fd);
    return (int)n;
}

static int parse_url_port(const char *url, int *port)
{
    *port = 8080;
    const char *p = strstr(url, "://");
    p = p ? p + 3 : url;
    const char *colon = strchr(p, ':');
    if (colon) *port = atoi(colon + 1);
    return 0;
}

static int probe_ready(const char *url)
{
    int port = 8080;
    parse_url_port(url, &port);
    char req[128];
    snprintf(req, sizeof(req), "GET /readyz HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n");
    char buf[1024];
    if (http_req("127.0.0.1", port, req, buf, sizeof(buf)) < 0) return 0;
    return strstr(buf, "200") != NULL;
}

static void draw(TuiState *s)
{
    write(STDOUT_FILENO, "\x1b[H\x1b[J", 6);
    dprintf(STDOUT_FILENO, "oxidize-c tui  %s  %s  %s\n", s->status, s->model_id, s->url);
    const char *tabs[] = {"chat", "models", "monitor", "logs"};
    for (int i = 0; i < 4; i++) {
        if (i == s->view) dprintf(STDOUT_FILENO, "[%s] ", tabs[i]);
        else dprintf(STDOUT_FILENO, " %s  ", tabs[i]);
    }
    dprintf(STDOUT_FILENO, "\n\n");
    if (s->overlay == 2) {
        dprintf(STDOUT_FILENO,
                "help: ctrl+k palette  ctrl+t view  1-4 jump  enter send/load\n"
                "esc cancel  q quit (not in chat)\n");
        return;
    }
    if (s->overlay == 1) {
        dprintf(STDOUT_FILENO, "palette > %s\n", s->palq);
        int shown = 0, idx = 0;
        for (size_t i = 0; i < sizeof(PAL) / sizeof(PAL[0]); i++) {
            if (!oc_tui_fuzzy(s->palq, PAL[i])) continue;
            dprintf(STDOUT_FILENO, "%s %s\n", idx == s->palcur ? ">" : " ", PAL[i]);
            idx++;
            shown++;
        }
        (void)shown;
        return;
    }
    if (s->view == VIEW_CHAT) {
        for (int i = 0; i < s->nturns; i++) {
            dprintf(STDOUT_FILENO, "user: %s\n", s->turns[i].user);
            dprintf(STDOUT_FILENO, "asst: %s\n\n", s->turns[i].asst);
        }
        dprintf(STDOUT_FILENO, "\n> %s\n", s->draft);
    } else if (s->view == VIEW_MODELS) {
        for (int i = 0; i < s->nmodels; i++) {
            dprintf(STDOUT_FILENO, "%s %-10lu %s\n",
                    i == s->mcursor ? ">" : " ",
                    s->models[i].size, s->models[i].name);
        }
        if (s->nmodels == 0) dprintf(STDOUT_FILENO, "(no .gguf under ./models or ~/.cache/oxidize/hf)\n");
    } else if (s->view == VIEW_MONITOR) {
        dprintf(STDOUT_FILENO, "%s\n", s->metrics[0] ? s->metrics : "(no /metrics yet)");
    } else {
        for (int i = 0; i < s->nlogs; i++) dprintf(STDOUT_FILENO, "%s\n", s->logs[i]);
    }
    dprintf(STDOUT_FILENO, "\nctrl+k palette  ctrl+t views  q quit\n");
}

static void scan_dir(TuiState *s, const char *dir, int depth)
{
    if (depth < 0 || s->nmodels >= 256) return;
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && s->nmodels < 256) {
        if (e->d_name[0] == '.') continue;
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) scan_dir(s, path, depth - 1);
        else {
            size_t n = strlen(e->d_name);
            if (n > 5 && oc_tui_fuzzy(".gguf", e->d_name + n - 5)) {
                TuiModel *m = &s->models[s->nmodels++];
                snprintf(m->name, sizeof(m->name), "%s", e->d_name);
                snprintf(m->path, sizeof(m->path), "%s", path);
                m->size = (unsigned long)st.st_size;
            }
        }
    }
    closedir(d);
}

static void scan_models(TuiState *s)
{
    s->nmodels = 0;
    scan_dir(s, "models", 3);
    const char *home = getenv("HOME");
    if (home) {
        char p[512];
        snprintf(p, sizeof(p), "%s/.cache/oxidize/hf", home);
        scan_dir(s, p, 4);
        snprintf(p, sizeof(p), "%s/models", home);
        scan_dir(s, p, 2);
    }
}

static int spawn_serve(TuiState *s, const char *model, const OcCliContext *ctx)
{
    int fds[2];
    if (pipe(fds) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        char *argv[16];
        int i = 0;
        argv[i++] = (char *)"oxidize-c";
        argv[i++] = (char *)"serve";
        argv[i++] = (char *)"--model";
        argv[i++] = (char *)model;
        argv[i++] = (char *)"--host";
        argv[i++] = (char *)"127.0.0.1";
        argv[i++] = (char *)"--port";
        argv[i++] = (char *)"8099";
        if (ctx->backend) {
            argv[i++] = (char *)"--backend";
            argv[i++] = (char *)ctx->backend;
        }
        argv[i] = NULL;
        execvp("oxidize-c", argv);
        execl("/proc/self/exe", "oxidize-c", "serve", "--model", model,
              "--host", "127.0.0.1", "--port", "8099", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    fcntl(fds[0], F_SETFL, O_NONBLOCK);
    s->child = pid;
    snprintf(s->url, sizeof(s->url), "http://127.0.0.1:8099");
    snprintf(s->status, sizeof(s->status), "starting");
    log_line(s, "spawned oxidize-c serve :8099");
    (void)fds[0];
    return 0;
}

static void json_escape(const char *in, char *out, size_t cap)
{
    size_t n = 0;
    for (; *in && n + 2 < cap; in++) {
        if (*in == '"' || *in == '\\') {
            if (n + 3 >= cap) break;
            out[n++] = '\\';
        }
        if (*in == '\n') {
            if (n + 3 >= cap) break;
            out[n++] = '\\';
            out[n++] = 'n';
            continue;
        }
        out[n++] = *in;
    }
    out[n] = '\0';
}

static void send_chat(TuiState *s)
{
    if (s->draft[0] == '\0' || s->generating) return;
    if (s->url[0] == '\0') {
        snprintf(s->status, sizeof(s->status), "no server — load a model");
        return;
    }
    if (s->nturns >= 32) return;
    TuiTurn *t = &s->turns[s->nturns++];
    snprintf(t->user, sizeof(t->user), "%s", s->draft);
    t->asst[0] = '\0';
    s->draft[0] = '\0';
    s->generating = 1;
    int port = 8080;
    parse_url_port(s->url, &port);
    char esc[4096];
    json_escape(t->user, esc, sizeof(esc));
    char body[4600];
    snprintf(body, sizeof(body),
             "{\"model\":\"%s\",\"stream\":true,\"temperature\":%.2f,"
             "\"max_tokens\":%d,\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
             s->model_id[0] ? s->model_id : "oxidize-default",
             s->temperature, s->max_tokens, esc);
    char req[5200];
    snprintf(req, sizeof(req),
             "POST /v1/chat/completions HTTP/1.0\r\nHost: 127.0.0.1\r\n"
             "Content-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
             strlen(body), body);
    char resp[16384];
    if (http_req("127.0.0.1", port, req, resp, sizeof(resp)) < 0) {
        snprintf(t->asst, sizeof(t->asst), "[error: request failed]");
        s->generating = 0;
        return;
    }
    char *line = resp;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        char delta[512];
        if (oc_tui_sse_delta(line, delta, sizeof(delta))) {
            size_t used = strlen(t->asst);
            snprintf(t->asst + used, sizeof(t->asst) - used, "%s", delta);
        }
        line = nl ? nl + 1 : NULL;
    }
    if (t->asst[0] == '\0') {
        char *bodyp = strstr(resp, "\r\n\r\n");
        snprintf(t->asst, sizeof(t->asst), "%s", bodyp ? bodyp + 4 : "[empty]");
    }
    s->generating = 0;
}

static void poll_metrics(TuiState *s)
{
    if (s->url[0] == '\0') return;
    int port = 8080;
    parse_url_port(s->url, &port);
    char req[96];
    snprintf(req, sizeof(req), "GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n");
    char buf[4096];
    if (http_req("127.0.0.1", port, req, buf, sizeof(buf)) < 0) return;
    char *body = strstr(buf, "\r\n\r\n");
    body = body ? body + 4 : buf;
    snprintf(s->metrics, sizeof(s->metrics), "tok/s %.1f  gen %.0f  inflight %.0f  errors %.0f",
             oc_tui_metric_value(body, "oxidize_tokens_per_second"),
             oc_tui_metric_value(body, "oxidize_tokens_generated_total"),
             oc_tui_metric_value(body, "oxidize_requests_in_flight"),
             oc_tui_metric_value(body, "oxidize_errors_total"));
}

static int read_key(char *seq, size_t cap)
{
    ssize_t n = read(STDIN_FILENO, seq, cap - 1);
    if (n <= 0) return 0;
    seq[n] = '\0';
    return (int)n;
}

static void run_palette(TuiState *s)
{
    int idx = 0;
    const char *chosen = NULL;
    for (size_t i = 0; i < sizeof(PAL) / sizeof(PAL[0]); i++) {
        if (!oc_tui_fuzzy(s->palq, PAL[i])) continue;
        if (idx == s->palcur) {
            chosen = PAL[i];
            break;
        }
        idx++;
    }
    s->overlay = 0;
    if (!chosen) return;
    if (strstr(chosen, "chat")) s->view = VIEW_CHAT;
    else if (strstr(chosen, "models")) s->view = VIEW_MODELS;
    else if (strstr(chosen, "monitor")) s->view = VIEW_MONITOR;
    else if (strstr(chosen, "logs")) s->view = VIEW_LOGS;
    else if (strstr(chosen, "Clear")) s->nturns = 0;
    else if (strstr(chosen, "Stop") && s->child > 0) {
        kill(s->child, SIGTERM);
        s->child = 0;
        snprintf(s->status, sizeof(s->status), "stopped");
    }
}

OcError oc_cli_run_tui(OcCliContext *ctx)
{
    if (!ctx) return OC_ERR_INVALID_ARG;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        fprintf(stderr, "oxidize-c tui needs a tty\n");
        return OC_ERR_INVALID_ARG;
    }
    TuiState s;
    memset(&s, 0, sizeof(s));
    s.temperature = ctx->temperature > 0 ? ctx->temperature : 0.8f;
    s.max_tokens = ctx->n_predict > 0 ? (int)ctx->n_predict : 512;
    snprintf(s.status, sizeof(s.status), "idle");
    snprintf(s.model_id, sizeof(s.model_id), "oxidize-default");
    scan_models(&s);
    if (ctx->attach_url) {
        snprintf(s.url, sizeof(s.url), "%s", ctx->attach_url);
        snprintf(s.status, sizeof(s.status), "attaching");
        s.view = VIEW_CHAT;
    } else if (ctx->model_path) {
        spawn_serve(&s, ctx->model_path, ctx);
        s.view = VIEW_CHAT;
    } else {
        s.view = VIEW_MODELS;
    }
    atexit(raw_off);
    raw_on();
    int ticks = 0;
    for (;;) {
        if (s.url[0] && strcmp(s.status, "ready") != 0 && probe_ready(s.url)) {
            snprintf(s.status, sizeof(s.status), "ready");
            log_line(&s, "server ready");
        }
        if ((ticks++ % 8) == 0) poll_metrics(&s);
        draw(&s);
        struct pollfd p = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&p, 1, 120) <= 0) continue;
        char seq[16];
        if (!read_key(seq, sizeof(seq))) continue;
        unsigned char c = (unsigned char)seq[0];
        if (c == 3) break; /* ctrl+c */
        if (s.overlay == 1) {
            if (c == 27) s.overlay = 0;
            else if (c == '\n' || c == '\r') run_palette(&s);
            else if (c == 127 || c == 8) {
                size_t n = strlen(s.palq);
                if (n) s.palq[n - 1] = '\0';
            } else if (c >= 32 && c < 127) {
                size_t n = strlen(s.palq);
                if (n + 1 < sizeof(s.palq)) {
                    s.palq[n] = (char)c;
                    s.palq[n + 1] = '\0';
                }
            }
            continue;
        }
        if (s.overlay == 2) {
            if (c == 27 || c == 'q' || c == '\n') s.overlay = 0;
            continue;
        }
        if (c == 11) { /* ctrl+k */
            s.overlay = 1;
            s.palq[0] = '\0';
            s.palcur = 0;
            continue;
        }
        if (c == 20) { /* ctrl+t */
            s.view = (s.view + 1) % 4;
            continue;
        }
        if (c == 27) continue;
        if (s.view == VIEW_CHAT) {
            if (c == '\n' || c == '\r') send_chat(&s);
            else if (c == 127 || c == 8) {
                size_t n = strlen(s.draft);
                if (n) s.draft[n - 1] = '\0';
            } else if (c >= 32 && c < 127) {
                size_t n = strlen(s.draft);
                if (n + 1 < sizeof(s.draft)) {
                    s.draft[n] = (char)c;
                    s.draft[n + 1] = '\0';
                }
            }
            continue;
        }
        if (c == 'q') break;
        if (c == '?') s.overlay = 2;
        if (c == '1') s.view = VIEW_CHAT;
        if (c == '2') s.view = VIEW_MODELS;
        if (c == '3') s.view = VIEW_MONITOR;
        if (c == '4') s.view = VIEW_LOGS;
        if (s.view == VIEW_MODELS) {
            if (c == 'j' || seq[0] == '\x1b') s.mcursor++;
            if (c == 'k' && s.mcursor > 0) s.mcursor--;
            if (s.mcursor >= s.nmodels && s.nmodels) s.mcursor = s.nmodels - 1;
            if ((c == '\n' || c == '\r') && s.nmodels) {
                spawn_serve(&s, s.models[s.mcursor].path, ctx);
                s.view = VIEW_CHAT;
            }
            if (c == 'r') scan_models(&s);
        }
        if (s.view == VIEW_LOGS && c == 'c') s.nlogs = 0;
    }
    raw_off();
    if (s.child > 0) {
        kill(s.child, SIGTERM);
        waitpid(s.child, NULL, 0);
    }
    return OC_OK;
}
