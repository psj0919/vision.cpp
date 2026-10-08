#include "util/math.h"
#include "util/string.h"
#include "visp/vision.h"
#include "visp/arch/ResNet.h"
#include "visp/arch/DetectionModel.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace visp {
using std::filesystem::path;

enum class cli_command { none, sam, birefnet, depth_anything, migan, esrgan, resnet, yolo };

struct cli_args {
    cli_command command = cli_command::none;
    std::vector<char const*> inputs;   // -i --input
    char const* output = "output.png"; // -o --output
    char const* model = nullptr;       // -m --model
    std::vector<char const*> prompt;   // -p --prompt
    // int threads = -1; // -t --threads
    // bool verbose = false; // -v --verbose
    bool gtx = false;
    bool log_ops = false;
    std::optional<backend_type> bknd_type; // -b --backend
    // std::string_view device = 0; // -d --device
    // ggml_type float_type = GGML_TYPE_COUNT; // -f32 -f16

    char const* composite = nullptr; // --composite
    char const* tensor_output = nullptr;
    char const* box_format = "cxcywh";
    int tile_size = -1;              // --tile
};

void print_usage() {
    char const* const usage = R"(
Usage: vision-cli <command> [options]

Commands:
    sam       - MobileSAM image segmentation
    birefnet  - BirefNet background removal
    depthany  - Depth-Anything depth estimation
    migan     - MI-GAN inpainting
    yolo      - YOLOv8n object detection (640x640 input)
    resnet    - ResNet18 classification (224x224 RGB, ImageNet normalization; -o logits text)
    esrgan    - ESRGAN/Real-ESRGAN upscaling

Options:
    -i, --input <image1> [<image2> ...]  Input image(s)
    -o, --output <file>                  Output file (default: output.png)
    -m, --model <file>                   Model file (.gguf)
    -p, --prompt <x> [<y> ...]           Prompt (eg. pixel coordinates)
    --log-ops                              Print graph operators before execution
    -b, --backend <cpu|gpu|gtx>              Backend type (default: auto)
    -h, --help                           Print usage and exit
    --composite <file>                   Composite input image with mask
    --dump-tensor <file>                 YOLO raw output tensor (binary)
    --box-format <cxcywh|xyxy>           YOLO box format (default: cxcywh)
    --tile <size>                        Tile size to split large images

Examples:
    vision-cli sam -m MobileSAM-F16.gguf -i image.jpg -p 100 200 -o mask.png
    vision-cli birefnet -m BiRefNet-F16.gguf -i image.jpg -o mask.png --composite output.png
    vision-cli migan -m MIGAN-F16.gguf -i image.jpg mask.png -o output.png
    vision-cli esrgan -m ESRGAN-x4-F16.gguf -i image.jpg -o upscaled.png
)";
    printf("%s", usage);
}

char const* const short_usage = R"(
Usage: vision-cli <command> [options]
See 'vision-cli --help' for more details.
)";

char const* next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw except("Missing argument after {}", argv[i - 1]);
    }
}

std::vector<char const*> collect_args(int argc, char** argv, int& i, char delim = '-') {
    std::vector<char const*> r;
    do {
        r.push_back(next_arg(argc, argv, i));
    } while (i + 1 < argc && argv[i + 1][0] != delim);
    if (r.empty()) {
        throw except("Missing argument after {}", argv[i - 1]);
    }
    return r;
}

int parse_int(std::string_view arg) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc()) {
        throw except("Invalid integer argument: {}", arg);
    }
    return value;
}

char const* validate_path(char const* arg) {
    if (!exists(path(arg))) {
        throw except("File not found: {}", arg);
    }
    return arg;
}

void require_inputs(std::span<char const* const> inputs, int n_required, char const* names) {
    if (inputs.size() != size_t(n_required)) {
        throw except(
            "Expected -i to be followed by {} inputs: {} - but found {}.", n_required, names,
            inputs.size());
    }
}

cli_args cli_parse(int argc, char** argv) {
    cli_args r;
    if (argc < 2) {
        throw except("Missing command.\n{}", short_usage);
    }

    std::string_view arg1 = argv[1];
    if (arg1 == "sam") {
        r.command = cli_command::sam;
    } else if (arg1 == "birefnet") {
        r.command = cli_command::birefnet;
    } else if (arg1 == "depthany" || arg1 == "depth-anything") {
        r.command = cli_command::depth_anything;
    } else if (arg1 == "migan") {
        r.command = cli_command::migan;
    } else if (arg1 == "yolo" || arg1 == "yolov8") {
        r.command = cli_command::yolo;
    } else if (arg1 == "resnet" || arg1 == "resnet18") {
        r.command = cli_command::resnet;
        r.output = "resnet-logits.txt";
    } else if (arg1 == "esrgan") {
        r.command = cli_command::esrgan;
    } else if (arg1 == "-h" || arg1 == "--help") {
        print_usage();
    } else {
        throw except("Unknown command: '{}'\n{}", arg1, short_usage);
    }

    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            r.inputs = collect_args(argc, argv, i);
            for_each(r.inputs.begin(), r.inputs.end(), validate_path);
        } else if (arg == "-o" || arg == "--output") {
            r.output = next_arg(argc, argv, i);
        } else if (arg == "-m" || arg == "--model") {
            r.model = next_arg(argc, argv, i);
        } else if (arg == "-p" || arg == "--prompt") {
            r.prompt = collect_args(argc, argv, i, '-');
        } else if (arg == "--log-ops") {
            r.log_ops = true;
        } else if (arg == "-b" || arg == "--backend") {
            std::string_view backend_arg = next_arg(argc, argv, i);
            r.gtx = false;
            r.bknd_type.reset();
            if (backend_arg == "cpu") {
                r.bknd_type = backend_type::cpu;
            } else if (backend_arg == "gtx") {
                r.gtx = true;
            } else if (backend_arg == "gpu") {
                r.bknd_type = backend_type::gpu;
            } else {
                throw except("Unknown backend type '{}', must be one of: cpu, gpu, gtx", backend_arg);
            }
        } else if (arg == "--composite") {
            r.composite = next_arg(argc, argv, i);
        } else if (arg == "--dump-tensor") {
            r.tensor_output = next_arg(argc, argv, i);
        } else if (arg == "--box-format") {
            r.box_format = next_arg(argc, argv, i);
            if (std::string_view(r.box_format) != "cxcywh" && std::string_view(r.box_format) != "xyxy")
                throw except("--box-format must be cxcywh or xyxy");
        } else if (arg == "--tile") {
            r.tile_size = parse_int(next_arg(argc, argv, i));
        } else if (arg.starts_with("-")) {
            throw except("Unknown argument: {}\n{}", arg, short_usage);
        }
    }
    return r;
}

void run_yolov8(cli_args const&);
void run_resnet(cli_args const&);
void run_sam(cli_args const&);
void run_birefnet(cli_args const&);
void run_depth_anything(cli_args const&);
void run_migan(cli_args const&);
void run_esrgan(cli_args const&);

} // namespace visp

//
// main

int main(int argc, char** argv) {
    using namespace visp;
    try {
        ggml_time_init();

        cli_args args = cli_parse(argc, argv);
        set_log_ops(args.log_ops);
        switch (args.command) {
            case cli_command::yolo: run_yolov8(args); break;
            case cli_command::resnet: run_resnet(args); break;
            case cli_command::sam: run_sam(args); break;
            case cli_command::birefnet: run_birefnet(args); break;
            case cli_command::depth_anything: run_depth_anything(args); break;
            case cli_command::migan: run_migan(args); break;
            case cli_command::esrgan: run_esrgan(args); break;
            case cli_command::none: break;
        }

    } catch (std::exception const& e) {
        printf("Error: %s\n", e.what());
        return 1;
    } catch (...) {
        return -1;
    }
    return 0;
}

namespace visp {

struct timer {
    int64_t start;
    fixed_string<16> string;

    timer() : start(ggml_time_us()) {}

    int64_t elapsed() const { return ggml_time_us() - start; }
    float elapsed_ms() const { return float(elapsed()) / 1000.0f; }

    char const* elapsed_str() {
        format(string, "{:.1f} ms", elapsed_ms());
        return string.c_str();
    }
};

//
// Common helpers

backend_device backend_init(cli_args const& args) {
    timer t;
    printf("Initializing backend... ");

    backend_device b;
    if (args.gtx) {
        char const* module = std::getenv("GGML_BACKEND_PATH");
        if (!module || !*module) throw except("-b gtx requires GGML_BACKEND_PATH pointing to libggml-gtx.so");
        auto reg = ggml_backend_load(module);
        if (!reg) throw except("Failed to load GTX backend library: {}", module);
        printf("loaded GTX backend from %s\n", module);
        auto device = ggml_backend_dev_by_name("GTX0");
        if (!device) throw except("GTX backend loaded but GTX0 device not registered");
        b.handle.reset(ggml_backend_dev_init(device, nullptr));
        if (!b.handle) throw except("GTX0 device initialization failed");
        b.device = device;
    } else if (args.bknd_type) {
        b = backend_init(*args.bknd_type);
    } else {
        b = backend_init();
    }
    printf("done (%s)\n", t.elapsed_str());

    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    char const* dev_name = ggml_backend_dev_name(dev);
    char const* dev_desc = ggml_backend_dev_description(dev);
    printf("- device: %s - %s\n", dev_name, dev_desc);
    return b;
}

char const* to_string(tensor_data_layout l) {
    switch (l) {
        case tensor_data_layout::cwhn: return "cwhn";
        case tensor_data_layout::whcn: return "whcn";
        default: return "unknown";
    }
}

path find_model(char const* model_name_or_path) {
    path p = path(model_name_or_path);
    if (exists(p) || p.is_absolute()) {
        return p;
    }
    path search_paths[5];
    search_paths[0] = path("models");
    if (char const* vision_model_dir = getenv("VISION_MODEL_DIR")) {
        search_paths[1] = path(vision_model_dir);
    }
    if (char const* xdg_data_home = getenv("XDG_DATA_HOME")) {
        search_paths[2] = path(xdg_data_home) / "visioncpp";
    }
    if (char const* home = getenv("HOME")) {
        search_paths[3] = path(home) / ".local/share/visioncpp";
    }
    if constexpr (VISP_MODEL_INSTALL_DIR[0] != '\0') {
        search_paths[4] = path(VISP_MODEL_INSTALL_DIR);
    }
    for (auto& sp : search_paths) {
        if (!sp.empty()) {
            path candidate = sp / p;
            if (exists(candidate)) {
                return candidate;
            }
        }
    }
    printf("Looking for %s\n", p.generic_string().c_str());
    for (auto& sp : search_paths) {
        if (!sp.empty()) {
            printf("Looking for %s\n", (sp / p).generic_string().c_str());
        }
    }
    throw except("Model file not found: {}", model_name_or_path);
}

std::tuple<model_file, model_weights> load_model_weights(
    cli_args const& args,
    backend_device const& dev,
    char const* default_model,
    int n_tensors = 0,
    tensor_data_layout preferred_layout = tensor_data_layout::unknown) {

    timer t;
    path model_path = find_model(args.model ? args.model : default_model);
    auto model_path_str = model_path.generic_string();
    printf("Loading model weights from '%s'... ", model_path_str.c_str());

    model_file file = model_load(model_path_str.c_str());
    model_weights weights = model_init(file.n_tensors() + n_tensors);
    if (preferred_layout == tensor_data_layout::unknown) {
        preferred_layout = file.tensor_layout();
    }
    model_transfer(file, weights, dev, dev.preferred_float_type(), preferred_layout);
    printf("done (%s)\n", t.elapsed_str());

    ggml_type ftype = file.float_type();
    if (ftype == GGML_TYPE_COUNT) {
        ftype = weights.float_type();
    }
    printf("- float type: %s\n", ggml_type_name(ftype));
    if (preferred_layout != tensor_data_layout::unknown) {
        printf("- tensor layout: %s\n", to_string(preferred_layout));
    }
    return {std::move(file), std::move(weights)};
}

void print_model_flags(model_ref const& m) {
    bool flash_attn = !!(m.flags & model_build_flag::flash_attention);
    printf("- flash attention: %s\n", flash_attn ? "on" : "off");
}

void compute_timed(compute_graph const& g, backend_device const& b) {
    timer t;
    printf("Running inference... ");
    compute(g, b);
    printf("complete (%s)\n", t.elapsed_str());
}

void composite_image_with_mask(image_view image, image_view mask, char const* output_path) {
    if (!output_path) {
        return;
    }
    image_data image_f32_data;
    if (!is_float(image.format)) {
        image_f32_data = image_u8_to_f32(image, image_format::rgba_f32);
        image = image_f32_data;
    }
    image_data mask_f32_data;
    if (!is_float(mask.format)) {
        mask_f32_data = image_u8_to_f32(mask, image_format::alpha_f32);
        mask = mask_f32_data;
    }

    image_data foreground = image_estimate_foreground(image, mask);

    image_data output = image_f32_to_u8(foreground, image_format::rgba_u8);
    image_save(output, output_path);
    printf("-> image composited and saved to %s\n", output_path);
}

//
// SAM

struct sam_prompt {
    i32x2 point1 = {-1, -1};
    i32x2 point2 = {-1, -1};

    bool is_point() const { return point2[0] == -1 || point2[1] == -1; }
    bool is_box() const { return !is_point(); }
};

sam_prompt sam_parse_prompt(std::span<char const* const> args, i32x2 extent) {
    if (args.empty()) {
        throw except(
            "SAM requires a prompt with coordinates for a point or box"
            "eg. '--prompt 100 200' to pick the point at pixel (x=100, y=200)");
    }
    if (args.size() < 2 || args.size() > 4) {
        throw except(
            "Invalid number of arguments for SAM prompt. Expected 2 (point) or 4 (box) numbers, "
            "got {}",
            args.size());
    }
    i32x2 a{-1, -1};
    if (args.size() >= 2) {
        a = {parse_int(args[0]), parse_int(args[1])};
        if (a[0] < 0 || a[1] < 0 || a[0] >= extent[0] || a[1] >= extent[1]) {
            throw except("Invalid image coordinates: ({}, {})", a[0], a[1]);
        }
    }
    i32x2 b{-1, -1};
    if (args.size() == 4) {
        b = {parse_int(args[2]), parse_int(args[3])};
        if (b[0] < 0 || b[1] < 0 || b[0] >= extent[0] || b[1] >= extent[1]) {
            throw except("Invalid image coordinates: ({}, {})", b[0], b[1]);
        }
        if (a[0] >= b[0] || a[1] >= b[1]) {
            throw except("Invalid box coordinates: ({}, {}) to ({}, {})", a[0], a[1], b[0], b[1]);
        }
    }
    return sam_prompt{a, b};
};

void run_sam(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "MobileSAM-F16.gguf", 0, backend.preferred_layout());
    sam_params params{};

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    image_data image_data_ = sam_process_input(image, params);

    sam_prompt prompt = sam_parse_prompt(args.prompt, image.extent);
    f32x4 prompt_data = prompt.is_point()
        ? sam_process_point(prompt.point1, image.extent, params)
        : sam_process_box({prompt.point1, prompt.point2}, image.extent, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    tensor image_tensor = compute_graph_input(m, GGML_TYPE_F32, {3, 1024, 1024, 1}, "image");
    tensor point_tensor = compute_graph_input(m, GGML_TYPE_F32, {2, 2, 1, 1}, "points");

    tensor image_embed = sam_encode_image(m, image_tensor, params);
    tensor prompt_embed = prompt.is_point() ? sam_encode_points(m, point_tensor)
                                            : sam_encode_box(m, point_tensor);

    sam_prediction output = sam_predict_mask(m, image_embed, prompt_embed);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(image_tensor, image_data_);
    transfer_to_backend(point_tensor, span(prompt_data.v, 4));

    compute_timed(graph, backend);

    timer t_post;
    printf("Postprocessing output... ");

    tensor_data iou = transfer_from_backend(output.iou);
    tensor_data mask_data = transfer_from_backend(output.masks);

    image_data mask = sam_process_mask(mask_data.as_f32(), 2, image.extent, params);
    printf("complete (%s)\n", t_post.elapsed_str());

    image_save(mask, args.output);

    auto ious = iou.as_f32();
    printf("-> estimated accuracy (IoU): %f, %f, %f\n", ious[0], ious[1], ious[2]);
    printf("-> mask saved to %s\n", args.output);

    composite_image_with_mask(image, mask, args.composite);
}

//
// BirefNet

void run_birefnet(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "BiRefNet-lite-F16.gguf", 0, backend.preferred_layout());

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    birefnet_params params = birefnet_detect_params(file, image.extent, backend.max_alloc());
    image_data input_data = birefnet_process_input(image, params);

    i32x2 extent = params.image_extent;
    char const* image_size_str = params.image_size < 0 ? " (dynamic)" : "";
    printf("- model image size: %d%s\n", params.image_size, image_size_str);
    printf("- inference image size: %dx%d\n", extent[0], extent[1]);

    compute_graph graph = compute_graph_init(6 * 1024);
    model_ref m(weights, graph);
    print_model_flags(m);

    birefnet_buffers buffers = birefnet_precompute(m, params);
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, extent[0], extent[1], 1});
    tensor output = birefnet_predict(m, input, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, input_data);
    for (tensor_data const& buf : buffers) {
        transfer_to_backend(buf);
    }

    compute_timed(graph, backend);

    tensor_data mask_data = transfer_from_backend(output);
    image_view mask_output(extent, mask_data.as_f32());
    image_data mask_resized = image_scale(mask_output, image.extent);
    image_data mask = image_f32_to_u8(mask_resized, image_format::alpha_u8);
    image_save(mask, args.output);
    printf("-> mask saved to %s\n", args.output);

    composite_image_with_mask(image, mask_resized, args.composite);
}

//
// Depth Anything

void run_depth_anything(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "DepthAnythingV2-Small-F32.gguf", 0, backend.preferred_layout());

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    depthany_params params = depthany_detect_params(file, image.extent);
    image_data input_data = depthany_process_input(image, params);

    i32x2 extent = params.image_extent;
    printf("- model image size: %d\n", params.image_size);
    printf("- inference image size: %dx%d\n", params.image_extent[0], params.image_extent[1]);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);
    print_model_flags(m);

    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, extent[0], extent[1], 1});
    tensor output = depthany_predict(m, input, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, input_data);

    compute_timed(graph, backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data depth_raw = depthany_process_output(output_data.as_f32(), image.extent, params);
    image_data depth_image = image_f32_to_u8(depth_raw, image_format::alpha_u8);
    image_save(depth_image, args.output);
    printf("-> depth image saved to %s\n", args.output);
}

//
// MI-GAN

void run_migan(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "MIGAN-512-places2-F16.gguf", backend.preferred_layout());
    migan_params params = migan_detect_params(file);
    params.invert_mask = true; // -> inpaint opaque areas

    require_inputs(args.inputs, 2, "<image> <mask>");
    image_data image = image_load(args.inputs[0]);
    image_data mask = image_load(args.inputs[1]);
    if (mask.format != image_format::alpha_u8) {
        mask = image_to_mask(mask);
    }
    image_data input_data = migan_process_input(image, mask, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    i64x4 input_shape = {4, params.resolution, params.resolution, 1};
    tensor input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
    tensor output = migan_generate(m, input, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, input_data);

    compute_timed(graph, backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data output_image = migan_process_output(output_data.as_f32(), image.extent, params);
    image_data mask_resized = image_scale(mask, image.extent);
    image_data composited = image_alpha_composite(output_image, image, mask_resized);
    image_save(composited, args.output);
    printf("-> output image saved to %s\n", args.output);
}

//
// ESRGAN

void run_esrgan(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "RealESRGAN-x4.gguf", 0, backend.preferred_layout());
    esrgan_params params = esrgan_detect_params(file);
    printf("- scale: %dx\n", params.scale);
    printf("- block count: %d\n", params.n_blocks);

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    int tile_size = args.tile_size > 0 ? args.tile_size : 224;

    tile_layout tiles = tile_layout(image.extent, tile_size, 16);
    tile_layout tiles_out = tile_scale(tiles, params.scale);
    image_data input_tile = image_alloc(tiles.tile_size, image_format::rgb_f32);
    image_data output_tile = image_alloc(tiles_out.tile_size, image_format::rgb_f32);
    image_data output_image = image_alloc(image.extent * params.scale, image_format::rgb_f32);
    image_clear(output_image);

    compute_graph graph = compute_graph_init(esrgan_estimate_graph_size(params));
    model_ref m(weights, graph);

    i64x4 input_shape = {3, tiles.tile_size[0], tiles.tile_size[1], 1};
    tensor input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
    tensor output = esrgan_generate(m, input, params);

    compute_graph_allocate(graph, backend);

    timer total;
    printf(
        "Using tile size %d with %d overlap -> %dx%d tiles\n", //
        tile_size, tiles.overlap[0], tiles.n_tiles[0], tiles.n_tiles[1]);

    for (int t = 0; t < tiles.total(); ++t) {
        printf("\rRunning inference... tile %d of %d", t + 1, tiles.total());
        i32x2 tile_coord = tiles.coord(t);
        i32x2 tile_offset = tiles.start(tile_coord);

        image_u8_to_f32(image, input_tile, f32x4(0), f32x4(1), tile_offset);
        transfer_to_backend(input, input_tile);

        compute(graph, backend);

        transfer_from_backend(output, output_tile);
        tile_merge(output_tile, output_image, tile_coord, tiles_out);
    }
    printf("\rRunning inference... complete (%s)\n", total.elapsed_str());

    image_data output_u8 = image_f32_to_u8(output_image, image_format::rgba_u8);
    image_save(output_u8, args.output);
    printf("-> output image saved to %s\n", args.output);
}

} // namespace visp

namespace visp {
void run_resnet(cli_args const& args) {
    require_inputs(args.inputs, 1, "image");
    auto b = backend_init(args);
    auto [file, weights] = load_model_weights(args, b, "ResNet.gguf", 0, tensor_data_layout::whcn);
    auto params = ResNet_detect_params(file);
    auto graph = compute_graph_init();
    model_ref m(weights, graph);
    auto input = compute_graph_input(m, GGML_TYPE_F32, {3, 224, 224, 1});
    auto output = ResNet_forward(m, input, params);
    if (ggml_nelements(output) != 1000) {
        throw except("Expected 1000 ResNet18 logits, got {}", ggml_nelements(output));
    }
    auto image = image_load(args.inputs[0]);
    auto resized = image_scale(image, {224, 224});
    auto normalized = image_u8_to_f32(resized, image_format::rgb_f32,
        {-.485f, -.456f, -.406f, 0.f},
        {1.f / .229f, 1.f / .224f, 1.f / .225f, 1.f});
    printf("- preprocessing: RGB bilinear resize 224x224, ImageNet mean/std\n");
    compute_graph_allocate(graph, b);
    transfer_to_backend(input, normalized);
    timer t;
    compute(graph, b);
    printf("Inference complete (%s)\n", t.elapsed_str());
    auto result = transfer_from_backend(output);
    auto logits = result.as_f32();
    for (float value : logits) if (!std::isfinite(value)) throw except("Non-finite ResNet logit");
    std::ofstream stream(args.output);
    if (!stream) throw except("Cannot open logits output: {}", args.output);
    stream << std::setprecision(9);
    for (float value : logits) stream << value << '\n';
    stream.close();
    if (!stream) throw except("Failed writing logits output: {}", args.output);
    auto best = std::max_element(logits.begin(), logits.end());
    printf("Top-1 class index: %zu, logit: %.9g\n", size_t(best - logits.begin()), *best);
    printf("Saved %zu logits to %s\n", logits.size(), args.output);
}
} // namespace visp

namespace visp {
namespace {
void draw_rect_outline(image_span const& img, i32x2 top_left, i32x2 bottom_right, int thickness = 2) {
    int ch = n_channels(img.format);
    uint8_t* base = static_cast<uint8_t*>(img.data);
    auto set_px = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= img.extent[0] || y >= img.extent[1]) {
            return;
        }
        uint8_t* p = base + size_t(y) * img.stride + size_t(x) * ch;
        p[0] = 255;
        if (ch > 1) p[1] = 32;
        if (ch > 2) p[2] = 32;
        if (ch > 3) p[3] = 255;
    };
    for (int t = 0; t < thickness; ++t) {
        for (int x = top_left[0]; x <= bottom_right[0]; ++x) {
            set_px(x, top_left[1] + t);
            set_px(x, bottom_right[1] - t);
        }
        for (int y = top_left[1]; y <= bottom_right[1]; ++y) {
            set_px(top_left[0] + t, y);
            set_px(bottom_right[0] - t, y);
        }
    }
}

struct yolo_box {
    float x0, y0, x1, y1;
    float score;
    int cls;
};

float box_iou(yolo_box const& a, yolo_box const& b) {
    float ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    float ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    float iw = std::max(0.f, ix1 - ix0), ih = std::max(0.f, iy1 - iy0);
    float inter = iw * ih;
    float area_a = std::max(0.f, a.x1 - a.x0) * std::max(0.f, a.y1 - a.y0);
    float area_b = std::max(0.f, b.x1 - b.x0) * std::max(0.f, b.y1 - b.y0);
    float uni = area_a + area_b - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Per-class greedy NMS (O(n^2), fine for a few hundred candidate boxes).
std::vector<yolo_box> nms(std::vector<yolo_box> boxes, float iou_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](yolo_box const& a, yolo_box const& b) {
        return a.score > b.score;
    });
    std::vector<bool> removed(boxes.size(), false);
    std::vector<yolo_box> kept;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!removed[j] && boxes[i].cls == boxes[j].cls &&
                box_iou(boxes[i], boxes[j]) > iou_thresh) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

} // namespace

void run_yolov8(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "YOLOv8n-F16.gguf", 0, tensor_data_layout::whcn);
    DetectionModel_params params = DetectionModel_detect_params(file);

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);

    // The compiled graph is unrolled for a fixed 640x640 input (see GTX_Compiler
    // --input-shape); boxes are decoded in that pixel space and rescaled back to
    // the original image size below (plain resize, not letterboxed).
    constexpr int model_size = 640;
    image_data resized = image_scale(image, {model_size, model_size});
    image_data input_data = image_alloc({model_size, model_size}, image_format::rgb_f32);
    // image_load() for u8 sources (image-impl.h) already normalizes to [0,1]
    // internally, so scale here must be 1 -- every other caller in this file
    // (SAM, MIGAN, etc.) uses f32x4(1.f); this one was wrongly re-dividing by
    // 255, squashing pixel values into [0, 1/255] and feeding the model
    // near-black noise instead of the actual image.
    image_u8_to_f32(resized, input_data, f32x4(0.f), f32x4(1.f));

    float conf_thresh = 0.25f;
    if (!args.prompt.empty()) {
        conf_thresh = float(std::atof(args.prompt[0]));
    }

    compute_graph graph = compute_graph_init(8 * 1024);
    model_ref m(weights, graph);

    tensor image_tensor =
        compute_graph_input(m, GGML_TYPE_F32, {3, model_size, model_size, 1}, "image");
    tensor out = DetectionModel_forward(m, image_tensor, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(image_tensor, input_data);

    compute_timed(graph, backend);

    timer t_post;
    printf("Postprocessing output... ");
    tensor_data out_data = transfer_from_backend(out);
    if (args.tensor_output) tensor_save(out, args.tensor_output);
    // out is [1, n_anchors, n_channels, 1] (c, w, h, n) — channel-major: value
    // for (channel k, anchor a) lives at data[k * n_anchors + a], not
    // anchor-major. Channels are [cx, cy, w, h, class_0 .. class_{n_classes-1}].
    auto [c1, n_anchors64, n_ch64, n] = nelements(out);
    (void)c1;
    (void)n;
    int n_anchors = int(n_anchors64);
    int n_ch = int(n_ch64);
    int n_classes = n_ch - 4;
    std::span<float const> data = out_data.as_f32();
    printf(
        "[debug] out shape anchors=%d channels=%d n_classes=%d\n", n_anchors, n_ch, n_classes);
    auto value = [&](int channel, int anchor) { return data[size_t(channel) * n_anchors + anchor]; };
    {
        float max_score = -1e30f, min_score = 1e30f;
        float max_box = -1e30f, min_box = 1e30f;
        for (int a = 0; a < n_anchors; ++a) {
            for (int k = 0; k < n_classes; ++k) {
                float s = value(4 + k, a);
                max_score = std::max(max_score, s);
                min_score = std::min(min_score, s);
            }
            for (int k = 0; k < 4; ++k) {
                float bv = value(k, a);
                max_box = std::max(max_box, bv);
                min_box = std::min(min_box, bv);
            }
        }
        printf(
            "[debug] class score range [%f, %f], box coord range [%f, %f]\n", min_score,
            max_score, min_box, max_box);
    }

    std::vector<yolo_box> boxes;
    for (int a = 0; a < n_anchors; ++a) {
        int best_cls = -1;
        float best_score = 0.f;
        for (int k = 0; k < n_classes; ++k) {
            float score = value(4 + k, a);
            if (score > best_score) {
                best_score = score;
                best_cls = k;
            }
        }
        if (best_cls >= 0 && best_score >= conf_thresh) {
            float v0 = value(0, a), v1 = value(1, a), v2 = value(2, a), v3 = value(3, a);
            if (std::string_view(args.box_format) == "xyxy") {
                boxes.push_back({v0, v1, v2, v3, best_score, best_cls});
            } else {
                float cx = v0, cy = v1, bw = v2, bh = v3;
                boxes.push_back(
                    {cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, best_score, best_cls});
            }
        }
    }
    boxes = nms(std::move(boxes), 0.45f);
    printf("complete (%s) -> %zu detections\n", t_post.elapsed_str(), boxes.size());

    float sx = float(image.extent[0]) / model_size;
    float sy = float(image.extent[1]) / model_size;
    image_span canvas(image);
    for (yolo_box const& b : boxes) {
        i32x2 top_left = {int(b.x0 * sx), int(b.y0 * sy)};
        i32x2 bottom_right = {int(b.x1 * sx), int(b.y1 * sy)};
        draw_rect_outline(canvas, top_left, bottom_right);
        printf(
            "  class=%d score=%.3f box=(%d,%d)-(%d,%d)\n", b.cls, b.score, top_left[0],
            top_left[1], bottom_right[0], bottom_right[1]);
    }

    image_save(image, args.output);
    printf("-> image with detections saved to %s\n", args.output);
}

} // namespace visp
