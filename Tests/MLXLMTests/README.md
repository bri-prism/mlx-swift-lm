The two includes resources are for testing the MediaProcessing pipeline for correctness and validation

The video file was created via FFMPEG via

```
ffmpeg \
-f lavfi \
-i smptebars=size=1920x1080:rate=30:duration=5.0 \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-vf "drawtext=text='Frame\\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:alpha=0.5:box=1:boxborderw=4,drawtext=text='TC':x=(w-tw)/2:y=(lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:fontcolor=white:timecode='01\\:00\\:00\\:00':timecode_rate=(30)" \
-c:v libx264 \
-c:a aac \
-crf 23 \
-preset medium \
-pix_fmt yuv420p \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y 1080p_30.mov
```

and the audio only file 

```
ffmpeg \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-c:a aac \
-crf 23 \
-preset medium \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y audio_only.mov
```

## Packed Qwen3.5 checkpoint integration

`PrismHadamardLoaderTests` includes opt-in text and vision comparisons against the
Python runtime bundled with a `prism_hadamard_qwen35` checkpoint. Download the
checkpoint unchanged, then generate reference files in a Python environment with
its required MLX/VLM dependencies:

```sh
python scripts/prism_hadamard_reference.py --model /path/to/checkpoint --output /tmp/packed-reference
TEST_RUNNER_PRISM_HADAMARD_MODEL=/path/to/checkpoint \
TEST_RUNNER_PRISM_HADAMARD_REFERENCE=/tmp/packed-reference \
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation -only-testing:MLXLMTests/PrismHadamardLoaderTests \
  -configuration Release ENABLE_TESTABILITY=YES CODE_SIGNING_ALLOWED=NO
```

The tests compare greedy tokens and full-vocabulary logit KL for prefill and
cached decoding. Vision uses identical Python-prepared pixels and prompt IDs;
it tests model execution and factory processor selection, not independent
Swift tokenizer or image-preprocessing parity. Without the environment variables,
the checkpoint tests skip while the small loader validation tests still run.
