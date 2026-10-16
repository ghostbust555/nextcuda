/* sd.h: the image kind's bridge to stable-diffusion.cpp (ggml-cuda), the CUDA implementation the Qwen-Image engines
 * are built on. A flat C ABI over its C++ library (include/stable-diffusion.h): no struct of that header crosses into
 * Rust, so a change of its layout is a recompile of the bridge and nothing else. One context is one loaded model on
 * one GPU; it keeps its weights between pictures. Every call returns 0, or -1 with the reason in ns_sd_last_error().
 */
#ifndef NS_IMAGE_SD_H
#define NS_IMAGE_SD_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_sd ns_sd;

const char* ns_sd_last_error(void);
const char* ns_sd_version(void);

/* what a model is made of (NULL: not part of it); `backend` names the device ("cuda0": ggml's number, which is
 * nvidia-smi's with CUDA_DEVICE_ORDER=PCI_BUS_ID); `max_vram_gib` 0: what the card has free; `offload`: weights stay in
 * host memory and are staged to the card per graph (the 12 GB cards); `args` is "key=value;..." for what the bridge
 * does not name (threads=N, flash_attn=0|1, vae_tiling=0|1) */
int ns_sd_create(const char* diffusion, const char* llm, const char* llm_vision, const char* vae, const char* backend,
                 float max_vram_gib, int offload, const char* args, ns_sd** out);
void ns_sd_destroy(ns_sd* s);

/* one picture. `init` (RGB or RGBA 8-bit, init_w x init_h x init_c) with strength < 1: img2img; `refs`: reference
 * pictures an edit conditions on, the first the one changed; `lora` + `lora_scale`: applied for this picture.
 * The picture comes back as 8-bit pixels of out_c channels (3, or 4 with `rgba`) in memory ns_sd_free_image frees. */
int ns_sd_generate(ns_sd* s, const char* prompt, const char* negative, int width, int height, int steps, float cfg, int64_t seed,
                   const char* sampler, const char* scheduler, float flow_shift, int rgba, float strength,
                   const uint8_t* init, int init_w, int init_h, int init_c,
                   const uint8_t* const* refs, const int* ref_w, const int* ref_h, const int* ref_c, int n_refs,
                   const char* lora, float lora_scale, uint8_t** out, int* out_w, int* out_h, int* out_c);
void ns_sd_free_image(uint8_t* p);

/* stop the picture being made (ends the call that is making it with -1) */
void ns_sd_cancel(ns_sd* s);

/* step, steps, seconds of the step; called on the thread making the picture. NULL: none. */
typedef void (*ns_sd_progress)(int step, int steps, float seconds, void* user);
void ns_sd_set_progress(ns_sd_progress cb, void* user);
/* the library's log lines (level 0 debug .. 3 error), one string a call; NULL: stderr */
typedef void (*ns_sd_log)(int level, const char* text, void* user);
void ns_sd_set_log(ns_sd_log cb, void* user);

#ifdef __cplusplus
}
#endif
#endif
