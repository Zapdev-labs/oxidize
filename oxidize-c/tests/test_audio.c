/* test_audio.c — audio multimodal module tests. */
#include "framework.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "oxidize/audio.h"

/* ─── Config defaults ──────────────────────────────────────────────────── */

Test(audio, config_default)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    cr_assert_eq(cfg.sample_rate, 16000);
    cr_assert_eq(cfg.n_mel_bins, 80);
    cr_assert_eq(cfg.n_fft, 400);
    cr_assert_eq(cfg.hop_length, 160);
    cr_assert_eq(cfg.chunk_length, 0);
    cr_assert_eq(cfg.max_audio_seconds, 30);
}

Test(audio, init_valid)
{
    OcAudioEncoder enc;
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    OcError e = oc_audio_init(&enc, &cfg, 512, 4);
    cr_assert_eq(e, OC_OK);
    cr_assert(enc.initialized);
    cr_assert_eq(enc.hidden_dim, 512);
    cr_assert_eq(enc.n_layers, 4);
    cr_assert_eq(enc.config.sample_rate, 16000);
    oc_audio_free(&enc);
}

Test(audio, init_null)
{
    OcError e = oc_audio_init(NULL, &OC_AUDIO_CONFIG_DEFAULT, 512, 4);
    cr_assert_neq(e, OC_OK);
}

Test(audio, init_bad_config)
{
    OcAudioEncoder enc;
    OcAudioConfig bad = OC_AUDIO_CONFIG_DEFAULT;
    bad.sample_rate = 0;
    cr_assert_neq(oc_audio_init(&enc, &bad, 512, 4), OC_OK);

    OcAudioConfig bad2 = OC_AUDIO_CONFIG_DEFAULT;
    bad2.n_fft = 0;
    cr_assert_neq(oc_audio_init(&enc, &bad2, 512, 4), OC_OK);

    OcAudioConfig bad3 = OC_AUDIO_CONFIG_DEFAULT;
    bad3.hop_length = 0;
    cr_assert_neq(oc_audio_init(&enc, &bad3, 512, 4), OC_OK);
}

Test(audio, init_bad_hidden_dim)
{
    OcAudioEncoder enc;
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    cr_assert_neq(oc_audio_init(&enc, &cfg, 0, 4), OC_OK);
}

/* ─── Mel scale conversions ─────────────────────────────────────────────── */

Test(audio, hz_to_mel_zero)
{
    cr_assert_float_eq(oc_audio_hz_to_mel(0.0f), 0.0f, 1e-6f);
}

Test(audio, hz_to_mel_mel_to_hz_roundtrip)
{
    for (float hz = 100.0f; hz <= 8000.0f; hz += 500.0f) {
        float mel = oc_audio_hz_to_mel(hz);
        float back = oc_audio_mel_to_hz(mel);
        cr_assert_float_eq(back, hz, 1.0f, "roundtrip failed for hz=%f", hz);
    }
}

Test(audio, hz_to_mel_increasing)
{
    float m1 = oc_audio_hz_to_mel(500.0f);
    float m2 = oc_audio_hz_to_mel(1000.0f);
    float m3 = oc_audio_hz_to_mel(2000.0f);
    cr_assert(m1 < m2, "mel should increase with hz");
    cr_assert(m2 < m3, "mel should increase with hz");
}

/* ─── Hann window ──────────────────────────────────────────────────────── */

Test(audio, hann_window_symmetry)
{
    uint32_t n = 256;
    float *w = malloc(n * sizeof(float));
    cr_assert_eq(oc_audio_hann_window(w, n), OC_OK);

    /* A Hann window is symmetric: w[i] == w[n-1-i]. */
    for (uint32_t i = 0; i < n / 2; i++) {
        cr_assert_float_eq(w[i], w[n - 1 - i], 1e-5f,
                           "Hann window not symmetric at i=%u", i);
    }
    free(w);
}

Test(audio, hann_window_endpoints)
{
    /* Use odd n so the Hann window has an exact center sample = 1.0. */
    uint32_t n = 129;
    float *w = malloc(n * sizeof(float));
    cr_assert_eq(oc_audio_hann_window(w, n), OC_OK);

    /* Hann window endpoints should be ~0. */
    cr_assert_float_eq(w[0], 0.0f, 1e-6f);
    cr_assert_float_eq(w[n - 1], 0.0f, 1e-6f);

    /* Center should be close to 1. */
    cr_assert_float_eq(w[n / 2], 1.0f, 0.01f);
    free(w);
}

Test(audio, hann_window_single)
{
    float w;
    cr_assert_eq(oc_audio_hann_window(&w, 1), OC_OK);
    cr_assert_float_eq(w, 1.0f, 1e-6f);
}

Test(audio, hann_window_null)
{
    cr_assert_neq(oc_audio_hann_window(NULL, 10), OC_OK);
}

/* ─── DFT ───────────────────────────────────────────────────────────────── */

Test(audio, dft_dc_signal)
{
    /* A constant (DC) signal should have energy only at bin 0. */
    uint32_t n = 64;
    float input[64];
    for (uint32_t i = 0; i < n; i++) input[i] = 1.0f;

    float mag[33]; /* n/2 + 1 */
    cr_assert_eq(oc_audio_dft(input, n, mag), OC_OK);

    /* DC bin should be n (sum of all 1.0s). */
    cr_assert_float_eq(mag[0], (float)n, 0.5f);

    /* All other bins should be ~0. */
    for (uint32_t k = 1; k < 33; k++) {
        cr_assert(fabsf(mag[k]) < 1e-3f, "non-DC bin %u should be ~0, got %f", k, mag[k]);
    }
}

Test(audio, dft_pure_tone)
{
    /* A pure sinusoid at bin k should produce a peak at bin k. */
    uint32_t n = 128;
    uint32_t k = 4;
    float input[128];
    for (uint32_t i = 0; i < n; i++) {
        input[i] = cosf(2.0f * (float)M_PI * (float)k * (float)i / (float)n);
    }

    float mag[65];
    cr_assert_eq(oc_audio_dft(input, n, mag), OC_OK);

    /* The peak should be at bin k. */
    float max_val = 0.0f;
    uint32_t max_bin = 0;
    for (uint32_t b = 1; b < 65; b++) {
        if (mag[b] > max_val) {
            max_val = mag[b];
            max_bin = b;
        }
    }
    cr_assert_eq(max_bin, k, "DFT peak should be at bin %u, got %u", k, max_bin);
}

Test(audio, dft_null)
{
    cr_assert_neq(oc_audio_dft(NULL, 10, NULL), OC_OK);
}

/* ─── Mel filter bank ──────────────────────────────────────────────────── */

Test(audio, mel_filterbank_shape)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    uint32_t n_mels  = cfg.n_mel_bins;
    uint32_t n_freqs = cfg.n_fft / 2 + 1;

    float *fb = malloc((size_t)n_mels * n_freqs * sizeof(float));
    uint32_t out_freqs = 0;
    OcError e = oc_audio_mel_filterbank(&cfg, fb, &out_freqs);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(out_freqs, n_freqs, "n_freq_bins should be n_fft/2+1");

    free(fb);
}

Test(audio, mel_filterbank_nonneg)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    uint32_t n_mels  = cfg.n_mel_bins;
    uint32_t n_freqs = cfg.n_fft / 2 + 1;

    float *fb = malloc((size_t)n_mels * n_freqs * sizeof(float));
    oc_audio_mel_filterbank(&cfg, fb, NULL);

    /* All filter bank values should be >= 0. */
    for (uint32_t m = 0; m < n_mels; m++) {
        for (uint32_t k = 0; k < n_freqs; k++) {
            cr_assert(fb[m * n_freqs + k] >= 0.0f,
                      "filter bank value should be non-negative at m=%u, k=%u", m, k);
        }
    }
    free(fb);
}

Test(audio, mel_filterbank_triangular)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    uint32_t n_mels  = cfg.n_mel_bins;
    uint32_t n_freqs = cfg.n_fft / 2 + 1;

    float *fb = malloc((size_t)n_mels * n_freqs * sizeof(float));
    oc_audio_mel_filterbank(&cfg, fb, NULL);

    /* Each mel filter should have a peak value of 1.0 (triangular filters
     * peak at 1.0 at their center). At least one value should be > 0. */
    for (uint32_t m = 0; m < n_mels; m++) {
        bool has_nonzero = false;
        for (uint32_t k = 0; k < n_freqs; k++) {
            if (fb[m * n_freqs + k] > 1e-6f) has_nonzero = true;
            cr_assert(fb[m * n_freqs + k] <= 1.0f + 1e-6f,
                      "filter value should be <= 1.0");
        }
        cr_assert(has_nonzero, "mel filter %u should have at least one nonzero", m);
    }
    free(fb);
}

Test(audio, mel_filterbank_null)
{
    cr_assert_neq(oc_audio_mel_filterbank(NULL, NULL, NULL), OC_OK);
}

/* ─── Mel spectrogram ──────────────────────────────────────────────────── */

Test(audio, compute_mel_basic)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    /* Generate 1 second of audio at 16kHz. */
    size_t n_samples = cfg.sample_rate;
    float *samples = malloc(n_samples * sizeof(float));
    /* Sine wave at 440 Hz. */
    for (size_t i = 0; i < n_samples; i++) {
        samples[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)i /
                                 (float)cfg.sample_rate);
    }

    OcAudioFeatures feats;
    OcError e = oc_audio_compute_mel(&cfg, samples, n_samples, &feats);
    cr_assert_eq(e, OC_OK);

    cr_assert_eq(feats.n_mel_bins, cfg.n_mel_bins);
    cr_assert(feats.n_frames > 0, "should have frames");
    cr_assert(feats.mel_spectrogram != NULL);
    cr_assert(feats.duration_seconds > 0.0f);

    oc_audio_features_free(&feats);
    free(samples);
}

Test(audio, compute_mel_frame_count)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    /* n_samples = n_fft + 2 * hop → n_frames = 1 + 2 = 3 */
    size_t n_samples = cfg.n_fft + 2 * cfg.hop_length;
    float *samples = calloc(n_samples, sizeof(float));

    OcAudioFeatures feats;
    cr_assert_eq(oc_audio_compute_mel(&cfg, samples, n_samples, &feats), OC_OK);
    cr_assert_eq(feats.n_frames, 3);

    oc_audio_features_free(&feats);
    free(samples);
}

Test(audio, compute_mel_too_short)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    float samples[10] = {0};
    OcAudioFeatures feats;
    cr_assert_neq(oc_audio_compute_mel(&cfg, samples, 10, &feats), OC_OK);
}

Test(audio, compute_mel_null)
{
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    OcAudioFeatures feats;
    cr_assert_neq(oc_audio_compute_mel(&cfg, NULL, 1000, &feats), OC_OK);
    cr_assert_neq(oc_audio_compute_mel(NULL, (float[]){0}, 1000, &feats), OC_OK);
}

Test(audio, n_frames_utility)
{
    /* n_samples=1000, n_fft=400, hop=160 → (1000-400)/160 = 3, +1 = 4 */
    cr_assert_eq(oc_audio_n_frames(1000, 400, 160), 4);
    /* Too short. */
    cr_assert_eq(oc_audio_n_frames(100, 400, 160), 0);
    /* Exactly n_fft → 1 frame. */
    cr_assert_eq(oc_audio_n_frames(400, 400, 160), 1);
}

/* ─── Audio encoding ────────────────────────────────────────────────────── */

Test(audio, encode_no_weights_fallback)
{
    OcAudioEncoder enc;
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    oc_audio_init(&enc, &cfg, 256, 0);

    /* Create fake features: 80 mel bins x 4 frames. */
    uint32_t n_mels = 80, n_frames = 4;
    float *mel = malloc((size_t)n_mels * n_frames * sizeof(float));
    for (size_t i = 0; i < (size_t)n_mels * n_frames; i++) {
        mel[i] = (float)i * 0.01f;
    }
    OcAudioFeatures feats = {
        .mel_spectrogram = mel,
        .n_mel_bins = n_mels,
        .n_frames = n_frames,
        .duration_seconds = 1.0f,
    };

    float *out = malloc((size_t)n_frames * 256 * sizeof(float));
    size_t out_len = 0;
    OcError e = oc_audio_encode(&enc, &feats, out, &out_len);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(out_len, (size_t)n_frames * 256);

    /* Fallback: each hidden value is the mean of mel bins for that frame. */
    /* Frame 0: mel[0..79], mean = 0.01 * (0+1+...+79) / 80 */
    float expected_mean = 0.01f * (0.0f + 79.0f) / 2.0f;
    cr_assert_float_eq(out[0], expected_mean, 0.05f);

    free(mel);
    free(out);
    oc_audio_free(&enc);
}

Test(audio, encode_null)
{
    OcAudioEncoder enc;
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    oc_audio_init(&enc, &cfg, 256, 0);

    float out[256];
    size_t out_len;
    cr_assert_neq(oc_audio_encode(NULL, NULL, NULL, NULL), OC_OK);
    cr_assert_neq(oc_audio_encode(&enc, NULL, out, &out_len), OC_OK);

    oc_audio_free(&enc);
}

Test(audio, encode_mismatched_mel_bins)
{
    OcAudioEncoder enc;
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    oc_audio_init(&enc, &cfg, 256, 0);

    /* Features with wrong n_mel_bins. */
    float mel[4] = {0};
    OcAudioFeatures feats = {
        .mel_spectrogram = mel,
        .n_mel_bins = 4, /* != cfg.n_mel_bins (80) */
        .n_frames = 1,
        .duration_seconds = 0.0f,
    };
    float out[256];
    size_t out_len;
    cr_assert_neq(oc_audio_encode(&enc, &feats, out, &out_len), OC_OK);
    oc_audio_free(&enc);
}

/* ─── Prompt formatting ────────────────────────────────────────────────── */

Test(audio, format_prompt_basic)
{
    float emb[3] = {1.0f, 2.0f, 3.0f};
    char *prompt = NULL;
    OcError e = oc_audio_format_prompt(emb, 3, "describe the audio", &prompt);
    cr_assert_eq(e, OC_OK);
    cr_assert(prompt != NULL);

    /* Should start with <audio>. */
    cr_assert(strncmp(prompt, "<audio>", 7) == 0);
    /* Should contain the text. */
    cr_assert(strstr(prompt, "describe the audio") != NULL);
    /* Should end with the text. */
    cr_assert(strstr(prompt, "</audio>") != NULL);

    free(prompt);
}

Test(audio, format_prompt_null_text)
{
    float emb[2] = {0.5f, 0.5f};
    char *prompt = NULL;
    OcError e = oc_audio_format_prompt(emb, 2, NULL, &prompt);
    cr_assert_eq(e, OC_OK);
    cr_assert(prompt != NULL);
    cr_assert(strstr(prompt, "<audio>") != NULL);
    cr_assert(strstr(prompt, "</audio>") != NULL);
    free(prompt);
}

Test(audio, format_prompt_null)
{
    float emb[1] = {0};
    cr_assert_neq(oc_audio_format_prompt(NULL, 1, "text", NULL), OC_OK);
    cr_assert_neq(oc_audio_format_prompt(emb, 1, "text", NULL), OC_OK);
}

/* ─── Free functions ────────────────────────────────────────────────────── */

Test(audio, features_free)
{
    OcAudioFeatures feats = {
        .mel_spectrogram = malloc(80 * sizeof(float)),
        .n_mel_bins = 80,
        .n_frames = 1,
        .duration_seconds = 0.01f,
    };
    oc_audio_features_free(&feats);
    cr_assert(feats.mel_spectrogram == NULL);
    cr_assert_eq(feats.n_mel_bins, 0);
}

Test(audio, wav_free)
{
    OcAudioWav wav = {
        .samples = malloc(100 * sizeof(int16_t)),
        .n_samples = 100,
        .sample_rate = 16000,
        .n_channels = 1,
        .bits_per_sample = 16,
    };
    oc_audio_wav_free(&wav);
    cr_assert(wav.samples == NULL);
    cr_assert_eq(wav.n_samples, 0);
}

Test(audio, free_null_safe)
{
    /* Should not crash on NULL. */
    oc_audio_free(NULL);
    oc_audio_features_free(NULL);
    oc_audio_wav_free(NULL);
}

/* ─── WAV loading (non-existent file) ───────────────────────────────────── */

Test(audio, load_wav_nonexistent)
{
    OcAudioWav wav;
    OcError e = oc_audio_load_wav("/nonexistent/path/test.wav", &wav);
    cr_assert_eq(e, OC_ERR_IO);
}

Test(audio, load_wav_null)
{
    OcAudioWav wav;
    cr_assert_neq(oc_audio_load_wav(NULL, &wav), OC_OK);
    cr_assert_neq(oc_audio_load_wav("test.wav", NULL), OC_OK);
}

/* ─── WAV file creation + loading (temp file) ───────────────────────────── */

Test(audio, load_wav_valid)
{
    /* Create a minimal valid WAV file in /tmp. */
    const char *path = "./test_audio_tmp.wav";
    FILE *fp = fopen(path, "wb");
    cr_assert(fp != NULL, "should be able to create temp WAV file");

    uint16_t n_channels = 1;
    uint32_t sample_rate = 16000;
    uint16_t bits_per_sample = 16;
    uint32_t n_samples = 100;

    /* Generate sample data. */
    uint32_t data_size = n_samples * (bits_per_sample / 8) * n_channels;
    uint32_t byte_rate = sample_rate * n_channels * (bits_per_sample / 8);
    uint16_t block_align = n_channels * (bits_per_sample / 8);
    uint32_t chunk_size = 36 + data_size;

    /* RIFF header. */
    fwrite("RIFF", 1, 4, fp);
    uint8_t buf4[4];

    /* chunk_size (little-endian). */
    buf4[0] = chunk_size & 0xFF; buf4[1] = (chunk_size >> 8) & 0xFF;
    buf4[2] = (chunk_size >> 16) & 0xFF; buf4[3] = (chunk_size >> 24) & 0xFF;
    fwrite(buf4, 1, 4, fp);

    fwrite("WAVE", 1, 4, fp);

    /* fmt chunk. */
    fwrite("fmt ", 1, 4, fp);
    uint32_t fmt_size = 16;
    buf4[0] = fmt_size & 0xFF; buf4[1] = (fmt_size >> 8) & 0xFF;
    buf4[2] = (fmt_size >> 16) & 0xFF; buf4[3] = (fmt_size >> 24) & 0xFF;
    fwrite(buf4, 1, 4, fp);

    /* audio_format = 1 (PCM). */
    uint8_t buf2[2];
    buf2[0] = 1; buf2[1] = 0;
    fwrite(buf2, 1, 2, fp);

    /* n_channels. */
    buf2[0] = n_channels & 0xFF; buf2[1] = (n_channels >> 8) & 0xFF;
    fwrite(buf2, 1, 2, fp);

    /* sample_rate. */
    buf4[0] = sample_rate & 0xFF; buf4[1] = (sample_rate >> 8) & 0xFF;
    buf4[2] = (sample_rate >> 16) & 0xFF; buf4[3] = (sample_rate >> 24) & 0xFF;
    fwrite(buf4, 1, 4, fp);

    /* byte_rate. */
    buf4[0] = byte_rate & 0xFF; buf4[1] = (byte_rate >> 8) & 0xFF;
    buf4[2] = (byte_rate >> 16) & 0xFF; buf4[3] = (byte_rate >> 24) & 0xFF;
    fwrite(buf4, 1, 4, fp);

    /* block_align. */
    buf2[0] = block_align & 0xFF; buf2[1] = (block_align >> 8) & 0xFF;
    fwrite(buf2, 1, 2, fp);

    /* bits_per_sample. */
    buf2[0] = bits_per_sample & 0xFF; buf2[1] = (bits_per_sample >> 8) & 0xFF;
    fwrite(buf2, 1, 2, fp);

    /* data chunk. */
    fwrite("data", 1, 4, fp);
    buf4[0] = data_size & 0xFF; buf4[1] = (data_size >> 8) & 0xFF;
    buf4[2] = (data_size >> 16) & 0xFF; buf4[3] = (data_size >> 24) & 0xFF;
    fwrite(buf4, 1, 4, fp);

    /* Write sample data: simple sine. */
    for (uint32_t i = 0; i < n_samples; i++) {
        int16_t s = (int16_t)(10000 * sinf(2.0f * (float)M_PI * 440.0f *
                                           (float)i / (float)sample_rate));
        buf2[0] = s & 0xFF; buf2[1] = (s >> 8) & 0xFF;
        fwrite(buf2, 1, 2, fp);
    }

    fclose(fp);

    /* Load the WAV file. */
    OcAudioWav wav;
    OcError e = oc_audio_load_wav(path, &wav);
    cr_assert_eq(e, OC_OK, "should load valid WAV file");
    cr_assert_eq(wav.sample_rate, 16000);
    cr_assert_eq(wav.n_channels, 1);
    cr_assert_eq(wav.bits_per_sample, 16);
    cr_assert_eq(wav.n_samples, n_samples);
    cr_assert(wav.samples != NULL);

    /* Verify first sample value. */
    int16_t expected_first = (int16_t)(10000 * sinf(0.0f));
    cr_assert_eq(wav.samples[0], expected_first);

    oc_audio_wav_free(&wav);
    remove(path);
}

Test(audio, load_wav_bad_magic)
{
    const char *path = "/tmp/oxidize_test_bad_magic.wav";
    FILE *fp = fopen(path, "wb");
    cr_assert(fp != NULL);
    fwrite("XXXX", 1, 4, fp);
    fwrite("\x00\x00\x00\x00", 1, 4, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);
    fclose(fp);

    OcAudioWav wav;
    OcError e = oc_audio_load_wav(path, &wav);
    cr_assert_eq(e, OC_ERR_FORMAT);

    remove(path);
}

/* ─── Full pipeline: WAV -> mel -> encode ───────────────────────────────── */

Test(audio, pipeline_wav_to_mel)
{
    /* Create a WAV file with 1 second of audio. */
    const char *path = "./test_pipeline_tmp.wav";
    FILE *fp = fopen(path, "wb");
    cr_assert(fp != NULL);

    uint16_t n_channels = 1;
    uint32_t sample_rate = 16000;
    uint16_t bits_per_sample = 16;
    uint32_t n_samples = sample_rate; /* 1 second */
    uint32_t data_size = n_samples * 2;
    uint32_t byte_rate = sample_rate * 2;
    uint16_t block_align = 2;
    uint32_t chunk_size = 36 + data_size;

    fwrite("RIFF", 1, 4, fp);
    uint8_t b4[4]; uint8_t b2[2];
    b4[0]=chunk_size&0xFF; b4[1]=(chunk_size>>8)&0xFF; b4[2]=(chunk_size>>16)&0xFF; b4[3]=(chunk_size>>24)&0xFF; fwrite(b4,1,4,fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);
    b4[0]=16; b4[1]=0; b4[2]=0; b4[3]=0; fwrite(b4,1,4,fp);
    b2[0]=1; b2[1]=0; fwrite(b2,1,2,fp); /* PCM */
    b2[0]=n_channels&0xFF; b2[1]=0; fwrite(b2,1,2,fp);
    b4[0]=sample_rate&0xFF; b4[1]=(sample_rate>>8)&0xFF; b4[2]=0; b4[3]=0; fwrite(b4,1,4,fp);
    b4[0]=byte_rate&0xFF; b4[1]=(byte_rate>>8)&0xFF; b4[2]=(byte_rate>>16)&0xFF; b4[3]=(byte_rate>>24)&0xFF; fwrite(b4,1,4,fp);
    b2[0]=block_align&0xFF; b2[1]=0; fwrite(b2,1,2,fp);
    b2[0]=bits_per_sample&0xFF; b2[1]=0; fwrite(b2,1,2,fp);
    fwrite("data", 1, 4, fp);
    b4[0]=data_size&0xFF; b4[1]=(data_size>>8)&0xFF; b4[2]=(data_size>>16)&0xFF; b4[3]=(data_size>>24)&0xFF; fwrite(b4,1,4,fp);

    for (uint32_t i = 0; i < n_samples; i++) {
        int16_t s = (int16_t)(20000 * sinf(2.0f * (float)M_PI * 440.0f *
                                           (float)i / (float)sample_rate));
        b2[0] = s & 0xFF; b2[1] = (s >> 8) & 0xFF;
        fwrite(b2, 1, 2, fp);
    }
    fclose(fp);

    /* Load WAV. */
    OcAudioWav wav;
    cr_assert_eq(oc_audio_load_wav(path, &wav), OC_OK);

    /* Compute mel from WAV. */
    OcAudioConfig cfg = OC_AUDIO_CONFIG_DEFAULT;
    OcAudioFeatures feats;
    cr_assert_eq(oc_audio_compute_mel_from_wav(&cfg, &wav, &feats), OC_OK);
    cr_assert_eq(feats.n_mel_bins, cfg.n_mel_bins);
    cr_assert(feats.n_frames > 0);

    /* Encode. */
    OcAudioEncoder enc;
    cr_assert_eq(oc_audio_init(&enc, &cfg, 256, 0), OC_OK);
    float *hidden = malloc((size_t)feats.n_frames * 256 * sizeof(float));
    size_t hidden_len = 0;
    cr_assert_eq(oc_audio_encode(&enc, &feats, hidden, &hidden_len), OC_OK);
    cr_assert_eq(hidden_len, (size_t)feats.n_frames * 256);

    /* Format prompt. */
    char *prompt = NULL;
    cr_assert_eq(oc_audio_format_prompt(hidden, hidden_len, "what do you hear?", &prompt), OC_OK);
    cr_assert(strstr(prompt, "<audio>") != NULL);
    cr_assert(strstr(prompt, "what do you hear?") != NULL);

    free(prompt);
    free(hidden);
    oc_audio_free(&enc);
    oc_audio_features_free(&feats);
    oc_audio_wav_free(&wav);
    remove(path);
}
