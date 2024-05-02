# MLC Stable Diffusion for RK3588's Mali GPU 

Run Stable Diffusion on RK3588's Mali GPU with MLC/TVM.

Currently generate a 512x512 image costs about 500 seconds (including model loading and GPU kernel compilation time. The actual inference time is less). The U-Net runs at 21sec per iteration.

### Build

1. Get a Stable Diffusion 1.5 model from Hugging Face/CivitAI/whatever model site. You can use any fine-tuned model since all of them are based on the same architecture.

2. The model is likely in .safetensor/.pth format. Convert it to Hugging Face Diffusers format using `convert_model_from_pth_safetensors.py` script. You can use the following command to convert the model:

```shell
python ./convert_model_from_pth_safetensors.py --checkpoint_path ./anythingv5.safetensors --dump_path ./anythingv5/ --from_safetensors --original_config_file ./v1-inference.yaml
```
(This can be done on any machine, not necessarily on the RK3588)

3. Install TVM. Follow [TVM’s documentation](https://tvm.apache.org/docs/install/from_source.html) to build from latest master source. **Please enable the OpenCL backend**

4. You may need to check OpenCL support on your RK3588. Follow https://llm.mlc.ai/docs/install/gpu.html#orange-pi-5-rk3588-based-sbc .

5. Edit the `build.py` script to match the model you want to build. 

```python
def trace_models(
    device_str: str,
) -> Tuple[tvm.IRModule, Dict[str, List[tvm.nd.NDArray]]]:
    from diffusers import StableDiffusionPipeline

    #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionPipeline.from_pretrained("./anythingv5")
```
6. Build the model using the following command:

```shell
python ./build.py
```

The warning about missing tuned operators is expected. The build script will generate a `dist` directory with the built model.

7. Run the model using the following command:

```shell
python ./deploy.py --device-name opencl
```

## Tuning

1. Very recommended to tune the model on board (not using RPC) because the measurer will automatically detect a failed trial and run the next one. If you use RPC, the measurer will not detect the failed trial and will wait for timeout, which is set to ~3 minutes, causing the tuning even slower.

2. Do the workarounds of 3,4 in Pitfalls section.

3. Uncomment the code in `build.py` to enable tuning:

```python
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

    # # run tuning
    ms.relax_integration.tune_relax(
        mod=mod,
        target=args.target,
        params={},
        builder=ms.builder.LocalBuilder(
            max_workers=7,
            timeout_sec=450,
        ),
        runner=ms.runner.LocalRunner(timeout_sec=180,  # need to be that long!
                                     maximum_process_uses=1, # to avoid buggy behaivour of mali opencl that subsequent runs fail after the first failure # this code change is not committed yet
                                     evaluator_config=ms.runner.config.EvaluatorConfig(
                                            number=1,    # avoid timeout 2
                                            repeat=1,
                                            min_repeat_ms=0,  # https://github.com/apache/tvm/issues/16276
                                     )),
        # runner=runner,
        work_dir="log_db_my",
        max_trials_global=100000,
        max_trials_per_task=8000,
        seed=42,
        num_trials_per_iter=32,
    )
    os._exit(0)
```

4. Execute `build.py`. Wait for a long time(>20hrs, have a good day!). You can interrupt the tuning process at any time if you notice that the latency is not improving anymore. The tuning process will save the best result so far in the `log_db_my` directory. But the tuning process will not be able to continue from the last checkpoint if you interrupt it. (Please open an feature request in TVM if you want this feature. I want it too. See: https://discuss.tvm.apache.org/t/metaschedule-how-to-resume-tuning/15298). Better to run the tuning process in a screen session. 

- This is a script wrote by GPT-4 to plot the latency in the tuning process. you can use it to see if the tuning process is still improving the latency.
```python
import os
import re
import csv
import matplotlib.pyplot as plt

log_dir = 'log_db_fp16_clip_unet'

log_file_path = os.path.join(log_dir, 'logs/tvm.meta_schedule.logging.task_scheduler.log')


# 定义正则表达式模式
pattern_trials = r'Total trials: (\d+)'
pattern_latency = r'Total latency \(us\): ([\d\.e\+]+)'

# 初始化列表存储数据
trials_list = []
latency_list = []

# 打开日志文件
with open(log_file_path, 'r') as file:
    log_data = file.read()

# 匹配Total trials和Total latency
trials_matches = re.findall(pattern_trials, log_data)
latency_matches = re.findall(pattern_latency, log_data)

# 将匹配结果添加到列表中
for trial, latency in zip(trials_matches, latency_matches):
    trials_list.append(int(trial))
    latency_list.append(float(latency))

# 将数据写入CSV文件
with open('trials_latency.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Total trials', 'Total latency (us)'])
    for trial, latency in zip(trials_list, latency_list):
        writer.writerow([trial, latency])

# 绘制latency随trials的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(trials_list, latency_list, marker='')
plt.xlabel('Total trials')
plt.ylabel('Total latency (us)')
plt.title('Latency vs Trials')
plt.show()
```

5. Comment the tuning code in `build.py` and add a `MetaScheduleApplyDatabase` line to apply the result. The result applied first will not be replaced by the second one so you should apply the best result first and then the second best result.

6. Test your result and see if the model is faster, or, how many hours you have wasted waiting for tuning?

7. Happy hacking!

8. Remember to star this repo if you find it useful!

## Limitations

- Model is running in FP32. FP16 would be faster and smaller in memory but I don't know how to convert the model to FP16.
- The model is under-tuned because tuning it is so slow (The current result is from a 48-hour tuning). The model can be further optimized by tuning it for a longer time. But try FP16 first!

## Why not NPU? More FLOPS!

- The RKNPU2 SDK is crappy and buggy
- ~~RKNPU2 does not support MatMul >= 256x256 on its model convertion while U-Net has large MatMul operations.~~ Update: Since RKNPU2 SDK 2.0.0b0 the limitation is removed, so you can try to run the model on NPU. (currently I don't have the interest to do this. Better waiting for SD3 since its DiT architecture which is easier to add NPU as well as dynamic shape support?)

## Pitfalls

1. `torch._dynamo.exc.BackendCompilerFailed: backend='_capture' raised: AssertionError: Unsupported function type position_ids`
  downgrade torch to 2.0.0 ~ 2.1.1(2.1.1 tested working)

2. ` expect a Tuple with 1 elements,  but get a Tuple with 196 elements.`                                                                                 
  add a `[]` in utils.py:
  ```python
    def transform_params(
    ...
        #new_params[name] = vm[name + "_transform_params"](params)
        new_params[name] = vm[name + "_transform_params"]([params])
  ```
  Honestly I don't know why this is happening. Version mismatch?

3. `dmesg` shows `Iterator PROGRESS_TIMER timeout` error: Mali GPU timeout is too short. Increase the timeout in the Mali GPU driver:
    ```shell
    echo 99999999999 > /sys/class/misc/mali0/device/progress_timeout
    ```
4. `dmesg` shows `CS_INHERIT_FAULT` error: A GPU fault will sometimes cause subsequent GPU operations in the same process to fail. So when tuning the model, better run separate processes for each try: (in local_runners.py add maximum_process_uses=1 param to `PopenPoolExecutor`)
5. `CL_OUT_OF_HOST_MEMORY` error: See https://github.com/apache/tvm/issues/16276

## Original README

# Web Stable Diffusion

This project brings stable diffusion models onto web browsers. **Everything runs inside the browser with no server support.** To our knowledge, this is the world’s first stable diffusion completely running on the browser. Please checkout our [demo webpage](https://websd.mlc.ai/#text-to-image-generation-demo) to try it out.

You are also more than welcomed to checkout [Web LLM](https://github.com/mlc-ai/web-llm) if you are interested in deploying LLM-based chat bots to browser.

<img src="site/img/fig/browser-screenshot.png" alt="Browser screenshot"/>

We have been seeing amazing progress through AI models recently. Thanks to the open-source effort, developers can now easily compose open-source models together to produce amazing tasks. Stable diffusion enables the automatic creation of photorealistic images as well as images in various styles based on text input. These models are usually big and compute-heavy, which means we have to pipe through all computation requests to (GPU) servers when developing web applications based on these models. Additionally, most of the workloads have to run on a specific type of GPUs where popular deep-learning frameworks are readily available.

This project takes a step to change that status quo and bring more diversity to the ecosystem. There are a lot of reasons to get some (or all) of the computation to the client side. There are many possible benefits, such as cost reduction on the service provider side, as well as an enhancement for personalization and privacy protection. The development of personal computers (even mobile devices) is going in the direction that enables such possibilities. The client side is getting pretty powerful. 

Building special client apps for those applications is one option (which we also support), but won’t it be even more amazing if we can simply open a browser and directly bring AI natively to your browser tab? There is some level of readiness in the ecosystem. WebAssembly allows us to port more lower-level runtimes onto the web. To solve the compute problem, WebGPU is getting matured lately and enables native GPU executions on the browser.

We are just seeing necessary elements coming together on the client side, both in terms of hardware and browser ecosystem. Still, there are big hurdles to cross, to name a few:

* We need to bring the models somewhere without the relevant GPU-accelerated Python frameworks.
* Most of the AI frameworks have a heavy reliance on optimized computed libraries that are maintained by hardware vendors. We need to start from zero. To get the maximum benefit, we might also need to produce variants per client environment.
* Careful planning of memory usage so we can fit the models into memory.

We do not want to only do it for just one model. Instead, we would like to present a repeatable, hackable, composable workflow that enables anyone to easily develop and optimize these models in a **Python-first** environment and universally **deploy** them everywhere, including the web.

## Get Started

We have a [Jupyter notebook](https://github.com/mlc-ai/web-stable-diffusion/blob/main/walkthrough.ipynb) that walks you through all the stages, including

* elaborate the key points of web ML model deployment and how we do to meet these points,
* import the stable diffusion model,
* optimize the model,
* build the model,
* deploy the model locally with native GPU runtime, and
* deploy the model on web with WebGPU runtime.

If you want to go through these steps in command line, please follow the commands below:

<details><summary>Commands</summary>

* Install TVM Unity. You can either
    * use `pip3 install mlc-ai-nightly -f https://mlc.ai/wheels` to install the TVM Unity wheel, or
    * follow [TVM’s documentation](https://tvm.apache.org/docs/install/from_source.html) to build from source. **Please use `git checkout origin/unity` to checkout to TVM Unity after git clone.**
* To import, optimize and build the stable diffusion model:
    ```shell
    python3 build.py
    ```
    By default `build.py` takes `apple/m2-gpu` as build target. You can also specify CUDA target via
    ```shell
    python3 build.py --target cuda
    ```
* To deploy the model locally with native GPU runtime:
    ```shell
    python3 deploy.py --prompt "A photo of an astronaut riding a horse on mars."
    ```
    You can substitute the prompt with your own one, and optionally use `--negative-prompt "Your negative prompt"` to specify a negative prompt.
* To deploy the model on web with WebGPU runtime, the last section “Deploy on web” of the [walkthrough notebook](https://github.com/mlc-ai/web-stable-diffusion/blob/main/walkthrough.ipynb) has listed the full instructions which you can refer to. We also provide the same list of plain instructions here:
    <details><summary>Instructions</summary>

    First, let’s install all the prerequisite:
    1. [emscripten](https://emscripten.org). It is an LLVM-based compiler which compiles C/C++ source code to WebAssembly.
        - Follow the [installation instruction](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended) to install the latest emsdk.
        - Source `emsdk_env.sh` by `source path/to/emsdk_env.sh`, so that `emcc` is reachable from PATH and the command `emcc` works.
    2. [Rust](https://www.rust-lang.org/tools/install).
    3. [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/). It helps build Rust-generated WebAssembly, which used for tokenizer in our case here.
    4. Install jekyll by following the [official guides](https://jekyllrb.com/docs/installation/). It is the package we use for website.
    5. Install jekyll-remote-theme by command
        ```shell
        gem install jekyll-remote-theme
        ```
    6. Install [Chrome Canary](https://www.google.com/chrome/canary/). It is a developer version of Chrome that enables the use of WebGPU.

    We can verify the success installation by trying out `emcc`, `jekyll` and `wasm-pack` in terminal respectively.

    Then, prepare all the necessary dependencies for web build:
    ```shell
    ./scripts/prep_deps.sh
    ```

    We can now build the model to WebGPU backend and export the executable to disk in the WebAssembly file format, by running
    ```shell
    python3 build.py --target webgpu
    ```

    The last thing to do is setting up the site with
    ```shell
    ./scripts/local_deploy_site.sh
    ```

    With the site set up, you can go to `localhost:8888/` in Chrome Canary to try out the demo on your local machine. Don’t forget to use
    ```shell
    /Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary --enable-dawn-features=disable_robustness
    ```
    to launch Chrome Canary to turn off the robustness check from Chrome.
    </details>
</details>

## How?

The key technology here is machine learning compilation (MLC). Our solution is built on the shoulders of the open-source ecosystem, including PyTorch, Hugging Face diffusers and tokenizers, rust, wasm, and WebGPU. The main flow is built on Apache TVM Unity, an exciting ongoing development in the [Apache TVM](https://github.com/apache/tvm)

- We take [Runway’s stable diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) models from the Hugging Face diffuser library.
- We use [TorchDynamo](https://pytorch.org/tutorials/intermediate/dynamo_tutorial.html) and [Torch FX](https://pytorch.org/docs/stable/fx.html) to capture key model components into an IRModule in TVM.
- Each function in TVM’s IRModule can be further transformed and generated with runnable code that can be deployed universally on any environment supported by minimum TVM runtime (javascript being one of them).
- [TensorIR](https://arxiv.org/abs/2207.04296) and [MetaSchedule](https://arxiv.org/abs/2205.13603) are used to build automated solutions to generate optimized programs. These transformations are tuned on a specific device through native GPU runtimes and then used to generate optimized GPU shaders. We provide a database that records these transformations so new builds can be done without tuning.
- We build static memory planning optimizations to reuse memory across multiple layers.
- We use [Emscripten](https://emscripten.org/) and typescript to build a TVM web runtime that can deploy generated modules.
- We also leverage the [wasm port](https://blog.mithrilsecurity.io/porting-tokenizers-to-wasm/) of the [rust tokenizers library](https://github.com/huggingface/tokenizers) from Hugging Face.

![workflow](site/img/fig/workflow.svg)

All parts of this workflow are done in Python, except, of course, the last part which builds a 400-loc JavaScript app that connects things together. This is also a fun process of interactive development, bringing new models.

All these are made possible by the open-source ecosystem that we leverage. Specifically, we make heavy use of [TVM Unity](https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344), an exciting latest development in the TVM project that enables such Python-first interactive MLC development experiences which allows us to easily compose new optimizations, all in Python, and incrementally bring our app to the web. TVM Unity also provides an easy way to compose new solutions in the ecosystem. For example, we can bring in other WebGPU shader generators or shader libraries easily to this workflow in the future.

## Comparison with Native GPU Runtime, Limitations, and Opportunities

Besides the WebGPU runtime, we also provide options for native deployment with local GPU runtime. These options can be used both as a tool to deploy on a native environment as well as a reference point to compare native GPU driver performance and WebGPU.

WebGPU works by translating WGSL (WebGPU Shading Language) shaders to native shaders. So, in theory, we can reach zero gaps between the WebGPU runtime and the native environment. If we directly use Chrome to check the current demo on Apple silicon, however, we can find a performance degradation (about 3x). This is because Chrome’s WebGPU implementation inserts bound clips for all array index access, such that `a[i]` becomes `a[min(i, a.size)]`. Ideally, downstream shader compilers should be able to optimize the bound clipping out, but here unfortunately, it is not the case. This gap can be fixed once WebGPU implementation becomes more mature, checks the index access range, and drops such clipping.

You can get around this by using a special flag to launch Chrome (thanks to Dawn developers for providing the pointers), by exiting Chrome completely, then in the command line, type
```shell
/path/to/chrome-canary --enable-dawn-features=disable_robustness
```
Then you will find that the execution speed is as fast as the native GPU environment. We anticipate this problem will get resolved as WebGPU matures.

We are just seeing the dawn of what we believe to be an eruption. WebGPU is still evolving (though it is getting close to shipping this year), and only available through Chrome Canary, and can be unstable. It also still comes with limitations, such as only support for FP32 (FP16 shader extension is on the spec but not yet implemented). The stable diffusion here would require a GPU with a decent amount of RAM (8GB). We have only tested our solution through Apple silicons so far. There are also opportunities to support advanced optimizations such as [FlashAttention](https://arxiv.org/abs/2205.14135) and quantization to further improve the performance of the system.

These are opportunities to bring several times of performance improvements to the current solutions. We believe many of them can be tackled in the near future. A single component of this solution can still be useful. For example, one can choose just to deploy the text encoder part of the model. Additionally, the same Python-first development, universal deployment workflow can be used to bring ML models to other environments, such as new hardware or mobile cases. Finally, the same machine learning compilation stack is also shared with server class use cases and can be used to optimize server workloads as well.

## Acknowledgement

This project is made possible thanks to collaboration with

<a href="https://www.scs.cmu.edu">
<img src="site/img/logo/cmuscs.png" alt="CMU School of Computer Science" height="50"/>
</a>
<a href="https://catalyst.cs.cmu.edu">
<img src="site/img/logo/catalyst.svg" alt="Catalyst" height="50"/>
</a>
<a href="https://mlc.ai">
<img src="site/img/logo/mlc-logo-with-text-landscape.svg" alt="MLC" height="50"/>
</a>
<a href="https://octoml.ai">
<img src="site/img/logo/octoml.png" alt="OctoML" height="50"/>
</a>

This project is only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort. We want to thank the open-source ML community members who make these models publicly available, and PyTorch, Hugging Face communities that make these models accessible. We would like to thank the tokenizer wasm port by Mithril Security. We also would like to thank the WebAssembly, Emscripten, Rust, and WebGPU communities. Finally, thanks to Dawn developers, who provide timely answers to questions on Chrome.
