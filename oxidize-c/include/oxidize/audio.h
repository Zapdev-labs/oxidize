#ifndef OXIDIZE_AUDIO_H
#define OXIDIZE_AUDIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct OcAudioConfig {
    uint32_t sample_rate;       /* e.g. 16000 Hz                          */
    uint32_t n_mel_bins;        /* mel filter bank count (e.g. 80)        */
    uint32_t n_fft;            /* FFT window size (e.g. 400)             */
    uint32_t hop_length;        /* hop between frames (e.g. 160)          */
    uint32_t chunk_length;      /* chunk size in samples (0 = no chunking) */
    uint32_t max_audio_seconds; /* hard cap on input length               */
} OcAudioConfig;

#define OC_AUDIO_CONFIG_DEFAULT ((OcAudioConfig){ \
    16000, 80, 400, 160, 0, 30 })

typedef struct OcAudioFeatures {
    float   *mel_spectrogram;   /* [n_mel_bins * n_frames]                */
    uint32_t n_mel_bins;        /* == config.n_mel_bins                    */
    uint32_t n_frames;         /* number of time frames                  */
    float    duration_seconds; /* n_frames * hop / sample_rate            */
} OcAudioFeatures;

typedef struct OcAudioEncoder {
    OcAudioConfig config;
    uint32_t      hidden_dim;   /* output embedding dimension            */
    uint32_t      n_layers;    /* encoder depth                        */
    /* Weight matrices (all optional; if NULL, uses fallback). */
    const float  *proj_weight;  /* [hidden_dim, n_mel_bins]              */
    const float  *proj_bias;    /* [hidden_dim]                         */
    const float  *pos_emb;      /* [max_n_frames, hidden_dim] (or NULL) */
    const float  *layer_weights;/* [n_layers, hidden_dim, hidden_dim]    */
    const float  *layer_biases; /* [n_layers, hidden_dim]                */
    bool          initialized;
} OcAudioEncoder;

typedef struct OcAudioWav {
    int16_t *samples;       /* mono 16-bit PCM (stereo averaged to mono) */
    size_t   n_samples;
    uint32_t sample_rate;
    uint16_t n_channels;
    uint16_t bits_per_sample;
} OcAudioWav;


/* Initialize the audio encoder with the given config. */
OcError oc_audio_init(OcAudioEncoder *enc, const OcAudioConfig *cfg,
                       uint32_t hidden_dim, uint32_t n_layers);

/* Free the audio encoder (resets to zero). */
void oc_audio_free(OcAudioEncoder *enc);

/* Free audio features (frees mel_spectrogram). */
void oc_audio_features_free(OcAudioFeatures *feats);

/* Free a parsed WAV file (frees samples). */
void oc_audio_wav_free(OcAudioWav *wav);

OcError oc_audio_load_wav(const char *path, OcAudioWav *out);

OcError oc_audio_mel_filterbank(const OcAudioConfig *cfg,
                                 float *out_filterbank,
                                 uint32_t *out_n_freq_bins);

OcError oc_audio_hann_window(float *out, uint32_t n);

OcError oc_audio_dft(const float *input, uint32_t n, float *out_mag);

OcError oc_audio_compute_mel(const OcAudioConfig *cfg,
                              const float *samples, size_t n_samples,
                              OcAudioFeatures *out_features);

/* Convenience: load WAV then compute mel spectrogram. */
OcError oc_audio_compute_mel_from_wav(const OcAudioConfig *cfg,
                                       const OcAudioWav *wav,
                                       OcAudioFeatures *out_features);

OcError oc_audio_encode(OcAudioEncoder *enc, const OcAudioFeatures *feats,
                         float *out_hidden, size_t *out_len);

OcError oc_audio_format_prompt(const float *audio_embeddings,
                                size_t n_embeddings,
                                const char *text,
                                char **out_prompt);

uint32_t oc_audio_n_frames(uint32_t n_samples, uint32_t n_fft,
                            uint32_t hop_length);

/* Convert frequency in Hz to mel scale (Slaney). */
float oc_audio_hz_to_mel(float hz);

/* Convert mel scale to frequency in Hz (Slaney). */
float oc_audio_mel_to_hz(float mel);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_AUDIO_H */
