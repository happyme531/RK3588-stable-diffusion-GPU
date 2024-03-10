from typing import Dict, List, Tuple

import os
import argparse
import pickle
import web_stable_diffusion.trace as trace
import web_stable_diffusion.utils as utils
from platform import system

import tvm
from tvm import relax


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--target", type=str, default="auto")
    args.add_argument("--db-path", type=str, default="log_db/")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--use-cache",
        type=int,
        default=1,
        help="Whether to use previously pickled IRModule and skip trace.",
    )
    args.add_argument("--debug-dump", action="store_true", default=False)

    parsed = args.parse_args()

    if parsed.target == "auto":
        if system() == "Darwin":
            target = tvm.target.Target("apple/m1-gpu")
        else:
            # has_gpu = tvm.cuda().exist
            # target = tvm.target.Target("cuda" if has_gpu else "llvm")
            target = tvm.target.Target("opencl -device=mali -max_shared_memory_per_block=32768 -max_threads_per_block=1024 -max_num_threads=1024 -thread_warp_size=16")
        print(f"Automatically configuring target: {target}")
        parsed.target = tvm.target.Target(target, host="llvm")
    elif parsed.target == "webgpu":
        parsed.target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )
    else:
        parsed.target = tvm.target.Target(parsed.target, host="llvm")
    return parsed


def debug_dump_script(mod, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w") as outfile:
        outfile.write(mod.script(show_meta=True))
    print(f"Dump mod to {dump_path}")


def debug_dump_shader(ex, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def trace_models(
    device_str: str,
) -> Tuple[tvm.IRModule, Dict[str, List[tvm.nd.NDArray]]]:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    clip = trace.clip_to_text_embeddings(pipe)
    unet = trace.unet_latents_to_noise_pred(pipe, device_str)
    vae = trace.vae_to_image(pipe)
    concat_embeddings = trace.concat_embeddings()
    image_to_rgba = trace.image_to_rgba()
    schedulers = [scheduler.scheduler_steps() for scheduler in trace.schedulers]

    mod = utils.merge_irmodules(
        clip,
        unet,
        vae,
        concat_embeddings,
        image_to_rgba,
        *schedulers,
    )
    # uncomment to use fp16 dtype for inference. 
    # will speed up the model but seems won't save memory (don't know why)
    # mod = relax.transform.ConvertToDataflow()(mod)
    # mod = relax.transform.ToMixedPrecision(out_dtype="float16")(mod)
    return relax.frontend.detach_params(mod)


def legalize_and_lift_params(
    mod: tvm.IRModule, model_params: Dict[str, List[tvm.nd.NDArray]], args: Dict
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    model_names = ["clip", "unet", "vae"]
    scheduler_func_names = [
        name
        for scheduler in trace.schedulers
        for name in scheduler.scheduler_steps_func_names()
    ]
    entry_funcs = (
        model_names + scheduler_func_names + ["image_to_rgba", "concat_embeddings"]
    )
    print(f"Entry functions: {entry_funcs}")

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.DeadCodeElimination(entry_funcs)(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod = relax.transform.BundleModelParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(
        mod, model_names, entry_funcs
    )

    debug_dump_script(mod_transform, "mod_lift_params.py", args)

    trace.compute_save_scheduler_consts(args.artifact_path)
    new_params = utils.transform_params(mod_transform, model_params)
    utils.save_params(new_params, args.artifact_path)
    return mod_deploy


def build(mod: tvm.IRModule, args: Dict) -> None:
    from tvm import meta_schedule as ms

    # # tuning part
    # # delete the VAE part of the model when tuning u-net. It will interfere with the tuning. Also it can run on NPU? https://clehaxze.tw/gemlog/2023/07-15-inexhaustive-list-of-models-that-works-on-rk3588.gmi
    # entry_funcs = ['clip', 'unet', 'dpm_solver_multistep_scheduler_convert_model_output', 'dpm_solver_multistep_scheduler_step', 'pndm_scheduler_step_0', 'pndm_scheduler_step_1', 'pndm_scheduler_step_2', 'pndm_scheduler_step_3', 'pndm_scheduler_step_4', 'image_to_rgba', 'concat_embeddings']
    # new_mod = tvm.IRModule()
    # for gv, func in mod.functions.items():
    #     try:
    #         if func.attrs["global_symbol"] == "main" and func.attrs["num_input"] == 1: # vae
    #             continue
    #     except:
    #         pass
    #     new_mod[gv] = func
    # mod = new_mod
    # mod = relax.transform.DeadCodeElimination(entry_funcs)(mod)
    # debug_dump_script(mod, "mod_tune.py", args)
    
    # # Important!! run `echo 99999999999 > /sys/class/misc/mali0/device/progress_timeout` before tuning to avoid timeout issue 1
    # # run tuning
    # ms.relax_integration.tune_relax(
    #     mod=mod,
    #     target=args.target,
    #     params={},
    #     builder=ms.builder.LocalBuilder(
    #         max_workers=7,
    #         timeout_sec=450,
    #     ),
    #     op_names={"softmax2"},
    #     runner=ms.runner.LocalRunner(timeout_sec=120,  # need to be that long!
    #                                  maximum_process_uses=1, # to avoid buggy behaivour of mali opencl that subsequent runs fail after the first failure # this code change is not committed yet
    #                                  evaluator_config=ms.runner.config.EvaluatorConfig(
    #                                         number=1,    # avoid timeout 2
    #                                         repeat=1,
    #                                         min_repeat_ms=0,  # https://github.com/apache/tvm/issues/16276
    #                                  )),
    #     work_dir="log_db_my",
    #     max_trials_global=100000,
    #     max_trials_per_task=8000,
    #     seed=42,
    #     num_trials_per_iter=32,
    # )
    # mydb = ms.database.create(work_dir="log_db_my")
    # mydb1 = ms.database.create(work_dir="log_db_my_pruned2_novae")
    # mydb.dump_pruned(
    #     mydb1,
    # )
    # db = ms.database.create(work_dir=args.db_path)
    # with args.target, mydb, tvm.transform.PassContext(opt_level=3):
    #     mod_deploy = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod)
    mod_deploy = mod
    print("Applying database 1 =======================")
    db3 = ms.database.create(work_dir="log_db_my_unet_softmax2") # For some reason, the softmax2 op run very slow on Mali GPU, so I need to tune it separately
    with args.target, db3, tvm.transform.PassContext(opt_level=3):
        mod_deploy = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod_deploy)
    print("Applying database 2 =======================")
    db0 = ms.database.create(work_dir="log_db_my_clip_unet") # The clip and unet part of the model
    with args.target, db0, tvm.transform.PassContext(opt_level=3):
        mod_deploy = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod_deploy)
    print("Applying database 3 =======================")
    db2 = ms.database.create(work_dir="log_db_my_vae") # The vae part of the model (Not tuned very well yet)
    with args.target, db2, tvm.transform.PassContext(opt_level=3):
        mod_deploy = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod_deploy)
    print("Generating missing schedules ==============")  
    with tvm.target.Target("cuda"):
        mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy) # for some missing schedules

    # i don't know why but the u-net, vae, clip symbol names changed to main and subgraph_0
    # get the original symbol names back
    # Delete this part if it is not necessary
    for gv, func in mod_deploy.functions.items():
        try:
            if func.attrs["global_symbol"] == "main" and func.attrs["num_input"] == 3: # u-net
                mod_deploy[gv] = func.with_attr("global_symbol", "unet")
            if func.attrs["global_symbol"] == "main" and func.attrs["num_input"] == 1: # vae
                mod_deploy[gv] = func.with_attr("global_symbol", "vae")
            if func.attrs["global_symbol"] == "subgraph_0":
                mod_deploy[gv] = func.with_attr("global_symbol", "clip")
        except:
            pass
    debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    ex = relax.build(mod_deploy, args.target)

    target_kind = args.target.kind.default_keys[0]

    if target_kind == "webgpu":
        output_filename = f"stable_diffusion_{target_kind}.wasm"
    else:
        output_filename = f"stable_diffusion_{target_kind}.so"

    debug_dump_shader(ex, f"stable_diffusion_{target_kind}", args)
    ex.export_library(os.path.join(args.artifact_path, output_filename))
    print(ex.stats())


if __name__ == "__main__":
    ARGS = _parse_args()
    os.makedirs(ARGS.artifact_path, exist_ok=True)
    os.makedirs(os.path.join(ARGS.artifact_path, "debug"), exist_ok=True)
    torch_dev_key = utils.detect_available_torch_device()
    cache_path = os.path.join(ARGS.artifact_path, "mod_cache_before_build.pkl")
    use_cache = ARGS.use_cache and os.path.isfile(cache_path)
    if not use_cache:
        mod, params = trace_models(torch_dev_key)
        mod = legalize_and_lift_params(mod, params, ARGS)
        with open(cache_path, "wb") as outfile:
            pickle.dump(mod, outfile)
        print(f"Save a cached module to {cache_path}.")
    else:
        print(
            f"Load cached module from {cache_path} and skip tracing. "
            "You can use --use-cache=0 to retrace"
        )
        mod = pickle.load(open(cache_path, "rb"))
    build(mod, ARGS)
