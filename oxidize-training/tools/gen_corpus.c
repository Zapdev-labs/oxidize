/* Emit a synthetic quant-trading corpus as UTF-8 lines.
 * Usage: gen_corpus [n_docs] [out_path]
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

/* Create every parent directory of `path`, like `mkdir -p $(dirname path)`. */
static int mkdir_parents(const char *path) {
    char buf[1024];
    if (snprintf(buf, sizeof buf, "%s", path) >= (int)sizeof buf) return -1;
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (buf[0] && mkdir(buf, 0777) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    return 0;
}

static unsigned rng = 1u;
static unsigned ru(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng;
}
static int ri(int lo, int hi) {
    return lo + (int)(ru() % (unsigned)(hi - lo + 1));
}
static const char *pick(const char **xs, int n) { return xs[ri(0, n - 1)]; }

static const char *TICKERS[] = {
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","AVGO","JPM",
    "XOM","UNH","LLY","COST","NFLX","CRM","ORCL","BAC","WMT",
    "SPY","QQQ","IWM","TLT","GLD","SMH","XLF"
};

static const char *TFS[] = {"5m","15m","1h","4h","1d"};

static const char *BUY_WHY[] = {
    "RSI reset while price held the prior breakout shelf",
    "a stop run through the overnight low was reclaimed on volume",
    "MACD crossed up with expanding histogram",
    "buyers defended VWAP twice and the tape turned bid",
    "the opening range broke high and held as support",
    "relative strength vs SPY stayed positive on the pullback"
};
static const char *SELL_WHY[] = {
    "a higher high printed on lower volume",
    "price failed the prior day high and trapped longs",
    "RSI is stretched and VWAP rejected twice",
    "MACD rolled over after the implied move was spent",
    "the opening range broke low and could not reclaim",
    "the name lagged SPY all session"
};
static const char *HOLD_WHY[] = {
    "the spread is wider than the stop",
    "event risk hits in under an hour",
    "two-way trade around VWAP has not chosen a side",
    "there is no edge versus SPY right now"
};

static const char *BUY_SETUP[] = {
    "trend pullback to the 20 EMA","opening range breakout",
    "failed breakdown reclaim","liquidity sweep then reverse"
};
static const char *SELL_SETUP[] = {
    "range fade at value area high","failed to take prior day high",
    "lower high on weak breadth","exhaustion at session highs"
};
static const char *HOLD_SETUP[] = {
    "balanced auction","inside day","news dead zone","wide spread, no follow"
};

static const char *TAIL_BUY[] = {
    "Size from the stop, not from hope.",
    "I will scratch if the first five minutes reverse.",
    "Scale half at plus one ATR, trail the rest.",
    "If SPY loses the opening range I flatten.",
};
static const char *TAIL_SELL[] = {
    "Cover half into the first flush.",
    "I will not short a reclaim of VWAP.",
    "If SPY rips I cover, no debate.",
    "Keep the stop tight. This is a fade, not a career.",
};
static const char *TAIL_HOLD[] = {
    "Cash is a position.",
    "Reassess on the next bar.",
    "Waiting is the trade.",
    "No fee today beats a forced tick.",
};

static void line_ticket(FILE *f) {
    const char *t = pick(TICKERS, (int)(sizeof TICKERS / sizeof *TICKERS));
    const char *tf = pick(TFS, (int)(sizeof TFS / sizeof *TFS));
    int side = (int)(ru() % 5u); /* 0-1 BUY, 2-3 SELL, 4 HOLD */
    int px = ri(20, 900);
    int atr = ri(2, 16);
    int rsi, stop, tgt, risk;
    risk = ri(25, 150);

    if (side <= 1) {
        rsi = ri(28, 48);
        stop = px - atr;
        tgt = px + 2 * atr;
        if (stop < 1) stop = 1;
        fprintf(f,
                "%s %s close %d RSI %d ATR %d. Setup: %s. ACTION=BUY because %s. "
                "Entry %d stop %d target %d. Risk $%d. %s\n",
                t, tf, px, rsi, atr, pick(BUY_SETUP, 4), pick(BUY_WHY, 6), px, stop, tgt, risk,
                pick(TAIL_BUY, 4));
    } else if (side <= 3) {
        rsi = ri(55, 82);
        stop = px + atr;
        tgt = px - 2 * atr;
        if (tgt < 1) tgt = 1;
        fprintf(f,
                "%s %s close %d RSI %d ATR %d. Setup: %s. ACTION=SELL because %s. "
                "Entry %d stop %d target %d. Risk $%d. %s\n",
                t, tf, px, rsi, atr, pick(SELL_SETUP, 4), pick(SELL_WHY, 6), px, stop, tgt, risk,
                pick(TAIL_SELL, 4));
    } else {
        rsi = ri(45, 58);
        fprintf(f,
                "%s %s close %d RSI %d ATR %d. Setup: %s. ACTION=HOLD because %s. %s\n",
                t, tf, px, rsi, atr, pick(HOLD_SETUP, 4), pick(HOLD_WHY, 4), pick(TAIL_HOLD, 4));
    }
}

static void line_macro(FILE *f) {
    static const char *m[] = {
        "Rising real yields pressure long-duration growth. Fade crowded momentum into CPI week.",
        "The dollar bid is fading. Cyclicals catch a bid. Gold is a hedge, not a hero.",
        "Credit spreads are quiet. Size half until HYG confirms.",
        "Vol is in backwardation. Sell the spike, do not chase it.",
        "Breadth is thinning into highs. Trade leaders, skip equal-weight beta.",
        "Oil inventory drew. Pair long XOM versus weak consumer names.",
    };
    fputs(pick(m, 6), f);
    fputc('\n', f);
}

static void line_lesson(FILE *f) {
    static const char *m[] = {
        "Position size from the stop, never from conviction.",
        "A trade without a stop is a donation.",
        "Do not add to a loser. Pyramids belong to winners.",
        "The opening range is a map. Volume confirms the break.",
        "Journal the reason before the fill. Memory will lie.",
        "Haircut the backtest for fees and skipped fills before you size up.",
    };
    fputs(pick(m, 6), f);
    fputc('\n', f);
}

int main(int argc, char **argv) {
    int n = 80000;
    if (argc > 1) {
        /* atoi turns malformed input into zero, which would silently become one doc. */
        char *end = NULL;
        errno = 0;
        long v = strtol(argv[1], &end, 10);
        if (errno == ERANGE || !end || end == argv[1] || *end || v < 1 || v > 100000000L) {
            fprintf(stderr, "usage: gen_corpus [n_docs] [out_path]\nn_docs must be 1..100000000\n");
            return 1;
        }
        n = (int)v;
    }
    const char *path = argc > 2 ? argv[2] : "corpus.txt";
    if (mkdir_parents(path) != 0) {
        perror(path);
        return 1;
    }
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        return 1;
    }
    rng = 42;
    for (int i = 0; i < n; i++) {
        int k = (int)(ru() % 10u);
        if (k < 8) line_ticket(f);
        else if (k < 9) line_macro(f);
        else line_lesson(f);
    }
    /* Buffered writes can fail as late as fclose, so check both. */
    int failed = ferror(f);
    if (fclose(f) != 0) failed = 1;
    if (failed) {
        perror(path);
        return 1;
    }
    fprintf(stderr, "wrote %d docs to %s\n", n, path);
    return 0;
}
