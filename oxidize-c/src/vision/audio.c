/* audio.c — Audio multimodal module implementation. */
#include "oxidize/audio.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define OC_MEL_F_SP        (200.0f / 3.0f)  /* ~66.667 */
#define OC_MEL_MIN_LOG_HZ  1000.0f
#define OC_MEL_MIN_LOG_MEL (1000.0f / OC_MEL_F_SP) /* 15.0 */
#define OC_MEL_LOGSTEP     0.06875177742094912f /* logf(6.4f) / 27.0f */

float oc_audio_hz_to_mel(float hz)
{
    if (hz < OC_MEL_MIN_LOG_HZ) {
        return hz / OC_MEL_F_SP;
    }
    return OC_MEL_MIN_LOG_MEL + logf(hz / OC_MEL_MIN_LOG_HZ) / OC_MEL_LOGSTEP;
}

float oc_audio_mel_to_hz(float mel)
{
    if (mel < OC_MEL_MIN_LOG_MEL) {
        return mel * OC_MEL_F_SP;
    }
    return OC_MEL_MIN_LOG_HZ * expf(OC_MEL_LOGSTEP * (mel - OC_MEL_MIN_LOG_MEL));
}


OcError oc_audio_init(OcAudioEncoder *enc, const OcAudioConfig *cfg,
                       uint32_t hidden_dim, uint32_t n_layers)
{
    if (!enc || !cfg) return OC_ERR_INVALID_ARG;
    if (cfg->sample_rate == 0 || cfg->n_mel_bins == 0 ||
        cfg->n_fft == 0 || cfg->hop_length == 0)
        return OC_ERR_INVALID_ARG;
    if (hidden_dim == 0) return OC_ERR_INVALID_ARG;

    memset(enc, 0, sizeof(*enc));
    enc->config     = *cfg;
    enc->hidden_dim = hidden_dim;
    enc->n_layers   = n_layers;
    enc->initialized = true;
    return OC_OK;
}

void oc_audio_free(OcAudioEncoder *enc)
{
    if (!enc) return;
    memset(enc, 0, sizeof(*enc));
}

void oc_audio_features_free(OcAudioFeatures *feats)
{
    if (!feats) return;
    free(feats->mel_spectrogram);
    memset(feats, 0, sizeof(*feats));
}

void oc_audio_wav_free(OcAudioWav *wav)
{
    if (!wav) return;
    free(wav->samples);
    memset(wav, 0, sizeof(*wav));
}


/* Read a little-endian 16-bit value from a byte buffer. */
static uint16_t oc_read_u16_le(const uint8_t *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

/* Read a little-endian 32-bit value from a byte buffer. */
static uint32_t oc_read_u32_le(const uint8_t *p)
{
    return (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

OcError oc_audio_load_wav(const char *path, OcAudioWav *out)
{
    if (!path || !out) return OC_ERR_INVALID_ARG;

    FILE *fp = fopen(path, "rb");
    if (!fp) return OC_ERR_IO;

    /* Read the first 12 bytes: RIFF + chunk_size + WAVE. */
    uint8_t riff_hdr[12];
    size_t nread = fread(riff_hdr, 1, sizeof(riff_hdr), fp);
    if (nread < 12) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }

    /* Verify RIFF header. */
    if (memcmp(riff_hdr, "RIFF", 4) != 0) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }
    if (memcmp(riff_hdr + 8, "WAVE", 4) != 0) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }

    /* Now scan subsequent chunks to find "fmt " and "data". */
    uint16_t audio_format    = 0;
    uint16_t n_channels      = 0;
    uint32_t sample_rate     = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size       = 0;
    bool found_fmt           = false;
    bool found_data          = false;

    uint8_t chunk_hdr[8];
    for (int i = 0; i < 64 && !found_data; i++) {
        if (fread(chunk_hdr, 1, 8, fp) != 8) {
            fclose(fp);
            return OC_ERR_FORMAT;
        }

        uint32_t chunk_size = oc_read_u32_le(chunk_hdr + 4);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            /* Read fmt data (at least 16 bytes). */
            uint8_t fmt_buf[40];
            uint32_t to_read = chunk_size < sizeof(fmt_buf) ? chunk_size : (uint32_t)sizeof(fmt_buf);
            if (fread(fmt_buf, 1, to_read, fp) != to_read) {
                fclose(fp);
                return OC_ERR_FORMAT;
            }
            /* Skip any remaining fmt bytes. */
            if (chunk_size > to_read) {
                if (fseek(fp, (long)(chunk_size - to_read), SEEK_CUR) != 0) {
                    fclose(fp);
                    return OC_ERR_IO;
                }
            }
            audio_format    = oc_read_u16_le(fmt_buf + 0);
            n_channels      = oc_read_u16_le(fmt_buf + 2);
            sample_rate     = oc_read_u32_le(fmt_buf + 4);
            bits_per_sample = oc_read_u16_le(fmt_buf + 14);
            found_fmt       = true;
        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            data_size  = chunk_size;
            found_data = true;
        } else {
            /* Unknown chunk — skip it. */
            if (fseek(fp, (long)chunk_size, SEEK_CUR) != 0) {
                fclose(fp);
                return OC_ERR_IO;
            }
        }
    }

    if (!found_fmt || !found_data) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }

    /* We support 16-bit PCM (audio_format == 1). */
    if (audio_format != 1 || bits_per_sample != 16) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }
    if (n_channels == 0) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }

    /* Read the data. */
    size_t n_bytes = data_size;
    size_t n_samples_total = n_bytes / 2; /* 16-bit = 2 bytes per sample */
    if (n_samples_total == 0) {
        fclose(fp);
        return OC_ERR_FORMAT;
    }

    uint8_t *raw = malloc(n_bytes);
    if (!raw) {
        fclose(fp);
        return OC_ERR_OOM;
    }

    size_t read_bytes = fread(raw, 1, n_bytes, fp);
    fclose(fp);

    if (read_bytes < n_bytes) {
        /* Truncated file; use what we have. */
        n_samples_total = read_bytes / 2;
        n_bytes = read_bytes;
    }

    /* De-interleave: convert to mono by averaging channels. */
    size_t n_frames_audio = n_samples_total / n_channels;
    int16_t *mono = malloc(n_frames_audio * sizeof(int16_t));
    if (!mono) {
        free(raw);
        return OC_ERR_OOM;
    }

    for (size_t i = 0; i < n_frames_audio; i++) {
        int32_t sum = 0;
        for (uint16_t ch = 0; ch < n_channels; ch++) {
            size_t idx = (i * n_channels + ch) * 2;
            int16_t val = (int16_t)(raw[idx] | (raw[idx + 1] << 8));
            sum += val;
        }
        mono[i] = (int16_t)(sum / n_channels);
    }

    free(raw);

    out->samples         = mono;
    out->n_samples       = n_frames_audio;
    out->sample_rate     = sample_rate;
    out->n_channels      = n_channels;
    out->bits_per_sample = bits_per_sample;
    return OC_OK;
}


OcError oc_audio_hann_window(float *out, uint32_t n)
{
    if (!out || n == 0) return OC_ERR_INVALID_ARG;
    if (n == 1) {
        out[0] = 1.0f;
        return OC_OK;
    }
    for (uint32_t i = 0; i < n; i++) {
        out[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1));
    }
    return OC_OK;
}


OcError oc_audio_dft(const float *input, uint32_t n, float *out_mag)
{
    if (!input || !out_mag || n == 0) return OC_ERR_INVALID_ARG;

    uint32_t out_len = n / 2 + 1;
    for (uint32_t k = 0; k < out_len; k++) {
        float re = 0.0f;
        float im = 0.0f;
        float angle_base = -2.0f * (float)M_PI * (float)k / (float)n;
        for (uint32_t j = 0; j < n; j++) {
            float angle = angle_base * (float)j;
            re += input[j] * cosf(angle);
            im += input[j] * sinf(angle);
        }
        /* Magnitude = sqrt(re^2 + im^2). We store power spectrum (magnitude^2)
         * in the mel computation step, but return magnitude here. */
        out_mag[k] = sqrtf(re * re + im * im);
    }
    return OC_OK;
}


OcError oc_audio_mel_filterbank(const OcAudioConfig *cfg,
                                 float *out_filterbank,
                                 uint32_t *out_n_freq_bins)
{
    if (!cfg || !out_filterbank) return OC_ERR_INVALID_ARG;

    uint32_t n_fft     = cfg->n_fft;
    uint32_t n_mels    = cfg->n_mel_bins;
    uint32_t n_freqs   = n_fft / 2 + 1; /* number of non-negative freq bins */

    if (out_n_freq_bins) *out_n_freq_bins = n_freqs;

    /* FFT bin frequencies: f[i] = i * sample_rate / n_fft */
    float fft_bin_hz = (float)cfg->sample_rate / (float)n_fft;

    /* Mel scale min/max. */
    float fmin_mel = oc_audio_hz_to_mel(0.0f);
    /* Use the Nyquist frequency as the max. */
    float fmax_hz  = (float)cfg->sample_rate / 2.0f;
    float fmax_mel = oc_audio_hz_to_mel(fmax_hz);

    /* Mel centers: n_mels + 2 points (inclusive) for triangular filters. */
    float *mel_points = malloc((n_mels + 2) * sizeof(float));
    if (!mel_points) return OC_ERR_OOM;

    for (uint32_t i = 0; i < n_mels + 2; i++) {
        mel_points[i] = fmin_mel + (fmax_mel - fmin_mel) * (float)i / (float)(n_mels + 1);
    }

    /* Convert mel points to Hz. */
    float *hz_points = malloc((n_mels + 2) * sizeof(float));
    if (!hz_points) { free(mel_points); return OC_ERR_OOM; }

    for (uint32_t i = 0; i < n_mels + 2; i++) {
        hz_points[i] = oc_audio_mel_to_hz(mel_points[i]);
    }

    /* Convert Hz to FFT bin indices. */
    float *bin_points = malloc((n_mels + 2) * sizeof(float));
    if (!bin_points) { free(mel_points); free(hz_points); return OC_ERR_OOM; }

    for (uint32_t i = 0; i < n_mels + 2; i++) {
        bin_points[i] = hz_points[i] / fft_bin_hz;
    }

    /* Build triangular filters. */
    memset(out_filterbank, 0, (size_t)n_mels * n_freqs * sizeof(float));

    for (uint32_t m = 0; m < n_mels; m++) {
        float left   = bin_points[m];
        float center = bin_points[m + 1];
        float right  = bin_points[m + 2];

        for (uint32_t k = 0; k < n_freqs; k++) {
            if ((float)k < left || (float)k > right) {
                out_filterbank[m * n_freqs + k] = 0.0f;
            } else if ((float)k <= center) {
                /* Rising edge. */
                float denom = center - left;
                if (denom > 1e-10f) {
                    out_filterbank[m * n_freqs + k] = ((float)k - left) / denom;
                }
            } else {
                /* Falling edge. */
                float denom = right - center;
                if (denom > 1e-10f) {
                    out_filterbank[m * n_freqs + k] = (right - (float)k) / denom;
                }
            }
        }
    }

    free(mel_points);
    free(hz_points);
    free(bin_points);
    return OC_OK;
}


uint32_t oc_audio_n_frames(uint32_t n_samples, uint32_t n_fft,
                           uint32_t hop_length)
{
    if (n_samples < n_fft || hop_length == 0) return 0;
    return 1 + (n_samples - n_fft) / hop_length;
}


OcError oc_audio_compute_mel(const OcAudioConfig *cfg,
                              const float *samples, size_t n_samples,
                              OcAudioFeatures *out_features)
{
    if (!cfg || !samples || !out_features) return OC_ERR_INVALID_ARG;
    if (cfg->n_fft == 0 || cfg->hop_length == 0 || cfg->n_mel_bins == 0)
        return OC_ERR_INVALID_ARG;

    /* Check max audio length. */
    if (cfg->max_audio_seconds > 0) {
        size_t max_samples = (size_t)cfg->max_audio_seconds * cfg->sample_rate;
        if (n_samples > max_samples) {
            n_samples = max_samples;
        }
    }

    uint32_t n_fft      = cfg->n_fft;
    uint32_t hop        = cfg->hop_length;
    uint32_t n_mels     = cfg->n_mel_bins;
    uint32_t n_freqs    = n_fft / 2 + 1;

    /* Compute number of frames. */
    if (n_samples < n_fft) {
        return OC_ERR_INVALID_ARG;
    }
    uint32_t n_frames = 1 + (uint32_t)((n_samples - n_fft) / hop);

    /* Allocate Hann window. */
    float *hann = malloc(n_fft * sizeof(float));
    if (!hann) return OC_ERR_OOM;
    oc_audio_hann_window(hann, n_fft);

    /* Allocate mel filter bank. */
    uint32_t fb_n_freqs = 0;
    float *filterbank = malloc((size_t)n_mels * n_freqs * sizeof(float));
    if (!filterbank) { free(hann); return OC_ERR_OOM; }

    OcError e = oc_audio_mel_filterbank(cfg, filterbank, &fb_n_freqs);
    if (e != OC_OK) {
        free(hann);
        free(filterbank);
        return e;
    }

    /* Allocate working buffers. */
    float *frame_buf = malloc(n_fft * sizeof(float));
    if (!frame_buf) { free(hann); free(filterbank); return OC_ERR_OOM; }

    float *power_spec = malloc(n_freqs * sizeof(float));
    if (!power_spec) { free(hann); free(filterbank); free(frame_buf); return OC_ERR_OOM; }

    /* Allocate output mel spectrogram: [n_mels x n_frames]. */
    float *mel_spec = malloc((size_t)n_mels * n_frames * sizeof(float));
    if (!mel_spec) {
        free(hann); free(filterbank); free(frame_buf); free(power_spec);
        return OC_ERR_OOM;
    }

    /* Process each frame. */
    for (uint32_t f = 0; f < n_frames; f++) {
        size_t offset = (size_t)f * hop;

        /* Extract frame and apply Hann window. */
        for (uint32_t i = 0; i < n_fft; i++) {
            frame_buf[i] = samples[offset + i] * hann[i];
        }

        /* Compute DFT magnitude. */
        oc_audio_dft(frame_buf, n_fft, power_spec);

        /* Convert to power spectrum (magnitude^2). */
        for (uint32_t k = 0; k < n_freqs; k++) {
            power_spec[k] = power_spec[k] * power_spec[k];
        }

        /* Apply mel filter bank. */
        for (uint32_t m = 0; m < n_mels; m++) {
            float mel_val = 0.0f;
            const float *fb_row = filterbank + (size_t)m * n_freqs;
            for (uint32_t k = 0; k < n_freqs; k++) {
                mel_val += fb_row[k] * power_spec[k];
            }
            /* Log compression (add small epsilon to avoid log(0)). */
            mel_val = logf(mel_val + 1e-10f);
            mel_spec[(size_t)m * n_frames + f] = mel_val;
        }
    }

    free(hann);
    free(filterbank);
    free(frame_buf);
    free(power_spec);

    out_features->mel_spectrogram = mel_spec;
    out_features->n_mel_bins      = n_mels;
    out_features->n_frames        = n_frames;
    out_features->duration_seconds = (float)n_frames * (float)hop /
                                      (float)cfg->sample_rate;
    return OC_OK;
}

OcError oc_audio_compute_mel_from_wav(const OcAudioConfig *cfg,
                                       const OcAudioWav *wav,
                                       OcAudioFeatures *out_features)
{
    if (!cfg || !wav || !out_features || !wav->samples) return OC_ERR_INVALID_ARG;
    if (wav->n_samples == 0) return OC_ERR_INVALID_ARG;

    /* Convert int16 samples to float [-1, 1]. */
    float *fbuf = malloc(wav->n_samples * sizeof(float));
    if (!fbuf) return OC_ERR_OOM;

    for (size_t i = 0; i < wav->n_samples; i++) {
        fbuf[i] = (float)wav->samples[i] / 32768.0f;
    }

    /* Override sample rate from WAV if config doesn't match. */
    OcAudioConfig effective = *cfg;
    effective.sample_rate = wav->sample_rate;

    OcError e = oc_audio_compute_mel(&effective, fbuf, wav->n_samples, out_features);
    free(fbuf);
    return e;
}


OcError oc_audio_encode(OcAudioEncoder *enc, const OcAudioFeatures *feats,
                         float *out_hidden, size_t *out_len)
{
    if (!enc || !enc->initialized || !feats || !feats->mel_spectrogram ||
        !out_hidden || !out_len)
        return OC_ERR_INVALID_ARG;
    if (feats->n_mel_bins != enc->config.n_mel_bins)
        return OC_ERR_INVALID_ARG;
    if (feats->n_frames == 0)
        return OC_ERR_INVALID_ARG;

    uint32_t hidden  = enc->hidden_dim;
    uint32_t n_mels  = enc->config.n_mel_bins;
    uint32_t n_frames = feats->n_frames;
    size_t total = (size_t)n_frames * hidden;

    if (enc->proj_weight != NULL) {
        /* Linear projection: [hidden, n_mels] x mel_vec[n_mels] -> hidden_vec[hidden]. */
        for (uint32_t t = 0; t < n_frames; t++) {
            const float *mel_vec = feats->mel_spectrogram + (size_t)t * n_mels;
            /* Note: mel_spectrogram is stored as [n_mels x n_frames],
             * so the frame index strides by 1 in the n_mels dimension.
             * We need to extract column t. */
            float *out_frame = out_hidden + (size_t)t * hidden;

            for (uint32_t h = 0; h < hidden; h++) {
                const float *w_row = enc->proj_weight + (size_t)h * n_mels;
                float dot = 0.0f;
                for (uint32_t m = 0; m < n_mels; m++) {
                    dot += w_row[m] * mel_vec[m];
                }
                if (enc->proj_bias) {
                    dot += enc->proj_bias[h];
                }
                out_frame[h] = dot;
            }

            /* Add positional embeddings if available. */
            if (enc->pos_emb) {
                const float *pos = enc->pos_emb + (size_t)t * hidden;
                for (uint32_t h = 0; h < hidden; h++) {
                    out_frame[h] += pos[h];
                }
            }
        }

        /* Apply simplified encoder layers (linear + GELU). */
        if (enc->layer_weights && enc->layer_biases && enc->n_layers > 0) {
            float *tmp = malloc(hidden * sizeof(float));
            if (!tmp) return OC_ERR_OOM;

            for (uint32_t layer = 0; layer < enc->n_layers; layer++) {
                const float *lw = enc->layer_weights +
                                  (size_t)layer * hidden * hidden;
                const float *lb = enc->layer_biases +
                                  (size_t)layer * hidden;

                for (uint32_t t = 0; t < n_frames; t++) {
                    float *frame = out_hidden + (size_t)t * hidden;
                    memcpy(tmp, frame, hidden * sizeof(float));

                    for (uint32_t h = 0; h < hidden; h++) {
                        float dot = 0.0f;
                        const float *w_row = lw + (size_t)h * hidden;
                        for (uint32_t j = 0; j < hidden; j++) {
                            dot += w_row[j] * tmp[j];
                        }
                        dot += lb[h];
                        /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
                        float x = dot;
                        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
                        frame[h] = 0.5f * x * (1.0f + tanhf(inner));
                    }
                }
            }
            free(tmp);
        }
    } else {
        /* Fallback: mean-pool mel bins across each frame (produces 1-D hidden
         * replicated, or just zeros for hidden_dim). Use mean of mel values
         * as a simple embedding. */
        for (uint32_t t = 0; t < n_frames; t++) {
            const float *mel_vec = feats->mel_spectrogram + (size_t)t * n_mels;
            float *out_frame = out_hidden + (size_t)t * hidden;
            float mean = 0.0f;
            for (uint32_t m = 0; m < n_mels; m++) {
                mean += mel_vec[m];
            }
            mean /= (float)n_mels;
            for (uint32_t h = 0; h < hidden; h++) {
                out_frame[h] = mean;
            }
        }
    }

    *out_len = total;
    return OC_OK;
}


OcError oc_audio_format_prompt(const float *audio_embeddings,
                                size_t n_embeddings,
                                const char *text,
                                char **out_prompt)
{
    if (!audio_embeddings || !out_prompt) return OC_ERR_INVALID_ARG;

    /* Estimate buffer size.
     * Format: "<audio>emb_0,emb_1,...,emb_n</audio>" + text
     * Each float: up to ~16 chars (sign + digits + decimal + exponent). */
    size_t per_float = 16;
    size_t emb_size  = n_embeddings * per_float;
    size_t text_len  = text ? strlen(text) : 0;
    /* "<audio>" (7) + ", " separators (2 * n) + "</audio>" (8) + text + NUL */
    size_t total = 7 + emb_size + 2 * n_embeddings + 8 + text_len + 1;

    char *buf = malloc(total);
    if (!buf) return OC_ERR_OOM;

    size_t pos = 0;
    /* Opening tag. */
    memcpy(buf + pos, "<audio>", 7);
    pos += 7;

    /* Embedding values. */
    for (size_t i = 0; i < n_embeddings; i++) {
        if (i > 0) {
            buf[pos++] = ',';
            buf[pos++] = ' ';
        }
        /* Write float with up to 6 decimal places. */
        int written = snprintf(buf + pos, total - pos, "%.6f", audio_embeddings[i]);
        if (written < 0) {
            free(buf);
            return OC_ERR_INTERNAL;
        }
        pos += (size_t)written;
    }

    /* Closing tag. */
    memcpy(buf + pos, "</audio>", 8);
    pos += 8;

    /* Append text. */
    if (text && text_len > 0) {
        memcpy(buf + pos, text, text_len);
        pos += text_len;
    }

    buf[pos] = '\0';
    *out_prompt = buf;
    return OC_OK;
}
