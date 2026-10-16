// sd.cpp: the bridge (sd.h) over stable-diffusion.cpp's C API.
#include "sd.h"

#include "stable-diffusion.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct ns_sd {
    sd_ctx_t* ctx = nullptr;
    // the strings the context's parameters pointed at while it was made
    std::vector<std::string> keep;
    std::map<std::string, std::string> args;
};

namespace {
thread_local std::string g_err;
int fail(const std::string& s) { g_err = s; return -1; }

ns_sd_progress g_progress = nullptr;
void* g_progress_user = nullptr;
ns_sd_log g_log = nullptr;
void* g_log_user = nullptr;

void on_progress(int step, int steps, float time, void*) {
    if (g_progress) g_progress(step, steps, time, g_progress_user);
}
void on_log(enum sd_log_level_t level, const char* text, void*) {
    if (g_log) { g_log((int) level, text, g_log_user); return; }
    if (level >= SD_LOG_WARN) fputs(text, stderr);
}

std::map<std::string, std::string> parse_args(const char* a) {
    std::map<std::string, std::string> m;
    std::string s = a ? a : "";
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(';', i);
        if (j == std::string::npos) j = s.size();
        const std::string kv = s.substr(i, j - i);
        const size_t e = kv.find('=');
        if (e != std::string::npos) m[kv.substr(0, e)] = kv.substr(e + 1);
        i = j + 1;
    }
    return m;
}
bool flag(const std::map<std::string, std::string>& m, const char* k, bool def) {
    auto it = m.find(k);
    return it == m.end() ? def : it->second != "0";
}
}  // namespace

extern "C" {

const char* ns_sd_last_error(void) { return g_err.c_str(); }
const char* ns_sd_version(void) { return "stable-diffusion.cpp (ggml-cuda)"; }

void ns_sd_set_progress(ns_sd_progress cb, void* user) {
    g_progress = cb;
    g_progress_user = user;
    sd_set_progress_callback(cb ? on_progress : nullptr, nullptr);
}

void ns_sd_set_log(ns_sd_log cb, void* user) {
    g_log = cb;
    g_log_user = user;
    sd_set_log_callback(on_log, nullptr);
}

int ns_sd_create(const char* diffusion, const char* llm, const char* llm_vision, const char* vae, const char* backend, float max_vram_gib,
                 int offload, const char* args, ns_sd** out) {
    auto* s = new ns_sd();
    s->args = parse_args(args);
    auto own = [&](const char* v) -> const char* {
        if (!v || !*v) return nullptr;
        s->keep.emplace_back(v);
        return s->keep.back().c_str();
    };
    // the strings must not move once pointed at: reserve first
    s->keep.reserve(16);
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.diffusion_model_path = own(diffusion);
    p.llm_path = own(llm);
    p.llm_vision_path = own(llm_vision);
    p.vae_path = own(vae);
    p.backend = own(backend);
    if (offload) p.params_backend = own("cpu");
    if (max_vram_gib > 0) {
        char b[32];
        std::snprintf(b, sizeof b, "%.2f", max_vram_gib);
        p.max_vram = own(b);
    }
    p.diffusion_flash_attn = flag(s->args, "flash_attn", true);
    p.flash_attn = p.diffusion_flash_attn;
    p.enable_mmap = true;
    auto t = s->args.find("threads");
    p.n_threads = t != s->args.end() ? atoi(t->second.c_str()) : sd_get_num_physical_cores();
    sd_set_log_callback(on_log, nullptr);
    s->ctx = new_sd_ctx(&p);
    if (!s->ctx) {
        delete s;
        return fail("stable-diffusion.cpp could not load the model (see its log)");
    }
    *out = s;
    return 0;
}

void ns_sd_destroy(ns_sd* s) {
    if (!s) return;
    if (s->ctx) free_sd_ctx(s->ctx);
    delete s;
}

int ns_sd_generate(ns_sd* s, const char* prompt, const char* negative, int width, int height, int steps, float cfg, int64_t seed,
                   const char* sampler, const char* scheduler, float flow_shift, int rgba, float strength, const uint8_t* init, int init_w,
                   int init_h, int init_c, const uint8_t* const* refs, const int* ref_w, const int* ref_h, const int* ref_c, int n_refs,
                   const char* lora, float lora_scale, uint8_t** out, int* out_w, int* out_h, int* out_c) {
    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt = prompt ? prompt : "";
    g.negative_prompt = negative ? negative : "";
    g.width = width;
    g.height = height;
    g.seed = seed;
    g.batch_count = 1;
    g.strength = strength;
    g.sample_params.sample_steps = steps;
    g.sample_params.guidance.txt_cfg = cfg;
    if (flow_shift > 0) g.sample_params.flow_shift = flow_shift;
    if (sampler && *sampler) {
        const auto m = str_to_sample_method(sampler);
        if (m == SAMPLE_METHOD_COUNT) return fail(std::string("sampler ") + sampler + " is not one stable-diffusion.cpp has");
        g.sample_params.sample_method = m;
    }
    if (scheduler && *scheduler) {
        const auto m = str_to_scheduler(scheduler);
        if (m == SCHEDULER_COUNT) return fail(std::string("scheduler ") + scheduler + " is not one stable-diffusion.cpp has");
        g.sample_params.scheduler = m;
    }
    g.vae_tiling_params.enabled = flag(s->args, "vae_tiling", true);
    if (init) g.init_image = sd_image_t{(uint32_t) init_w, (uint32_t) init_h, (uint32_t) init_c, const_cast<uint8_t*>(init)};
    std::vector<sd_image_t> r;
    for (int i = 0; i < n_refs; ++i) r.push_back(sd_image_t{(uint32_t) ref_w[i], (uint32_t) ref_h[i], (uint32_t) ref_c[i], const_cast<uint8_t*>(refs[i])});
    g.ref_images = r.empty() ? nullptr : r.data();
    g.ref_images_count = (int) r.size();
    // a `--disable-auto-resize-ref-image` of sd-cli: the engine above resizes references itself
    sd_lora_t l{false, lora_scale, lora};
    if (lora && *lora) {
        g.loras = &l;
        g.lora_count = 1;
    }
    sd_cancel_generation(s->ctx, SD_CANCEL_RESET);
    sd_image_t* imgs = nullptr;
    int n = 0;
    if (!generate_image(s->ctx, &g, &imgs, &n) || n < 1 || !imgs || !imgs[0].data) {
        return fail("stable-diffusion.cpp made no picture (see its log)");
    }
    const sd_image_t& im = imgs[0];
    const int c = im.channel;
    const size_t px = (size_t) im.width * im.height;
    // 3 or 4 channels out as asked: the library gives 3 (RGB), or 4 for an alpha prompt
    const int oc = rgba ? 4 : 3;
    auto* buf = (uint8_t*) std::malloc(px * oc);
    for (size_t i = 0; i < px; ++i) {
        for (int k = 0; k < 3; ++k) buf[i * oc + k] = im.data[i * c + (k < c ? k : c - 1)];
        if (oc == 4) buf[i * 4 + 3] = c == 4 ? im.data[i * 4 + 3] : 255;
    }
    *out = buf;
    *out_w = (int) im.width;
    *out_h = (int) im.height;
    *out_c = oc;
    for (int i = 0; i < n; ++i) std::free(imgs[i].data);
    std::free(imgs);
    return 0;
}

void ns_sd_free_image(uint8_t* p) { std::free(p); }

void ns_sd_cancel(ns_sd* s) {
    if (s && s->ctx) sd_cancel_generation(s->ctx, SD_CANCEL_ALL);
}

}  // extern "C"
