# DFlash2 on Qwen3.5 and Bonsai 2

`DFlash2DraftModel` loads the public BF16 DFlash2 checkpoint, including dynamic grouped convolutions, target-layer projections, and the candidate selector. `DFlash2TokenIterator` performs greedy, text-only speculative decoding against a Qwen3.5 target. The target can use the existing signed-Hadamard packed loader; no target weights are converted or replaced.

The initial implementation owns fresh caches. It verifies proposals against copies and replays only the accepted prefix after rejection, preserving the target's recurrent state. It does not resume external caches, accept image inputs, or implement stochastic rejection sampling. Context projection is recomputed each round; this is a correctness implementation, with no throughput guarantee.

## Usage

Load a target `ModelContext` using `LLMModelFactory` and your tokenizer loader, then prepare text with its processor. Download the drafter's `config.json` and `model.safetensors` into a separate directory.

```swift
import MLXLLM
import MLXLMCommon

let drafter = try DFlash2DraftModel.load(from: drafterDirectory)
let input = try await context.processor.prepare(
    input: UserInput(prompt: "Explain binary search."))
let iterator = try DFlash2TokenIterator(
    input: input.text.tokens,
    target: context.model,
    drafter: drafter,
    maxTokens: 256)
let (stream, task) = generateTask(
    promptTokenCount: input.text.tokens.size,
    modelConfiguration: context.configuration,
    tokenizer: context.tokenizer,
    iterator: iterator)
for await event in stream {
    if case .chunk(let text) = event { print(text, terminator: "") }
}
await task.value
```

The normal generation loop applies the target's stop tokens. Direct iterator users must handle EOS themselves. `proposedCount`, `acceptedCount`, `replayCount`, and `speculativeDecodingTelemetry` expose actual draft activity. Unsupported target geometry and unsupported drafter configurations throw rather than silently falling back to ordinary decoding.

## Reproduced checkpoints

- Target: [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/tree/3f926b415992eaa2ae9dd7b573706494d6bbf787).
- Drafter: [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b).

The drafter is an independent community adaptation. Checkpoint compatibility does not establish a performance benefit on every device or workload.

## Numerical references and tests

`DFlash2Tests` includes CPU-PyTorch-generated fixtures for dynamic convolution, candidate selection, and a complete two-layer forward pass with mixed full/sliding attention. Both fixture generators verify the hash of the public source before executing it. Source provenance is recorded in the fixture JSON files and the SpecForge license is included in `ThirdPartyLicenses/SpecForge.txt`.

With PyTorch, NumPy, safetensors and Transformers 5.5.0 installed:

```bash
python scripts/dflash2_reference.py \
  --source /path/to/extracted/specforge/modeling/draft/dflash2.py \
  --output Tests/MLXLMTests/Resources/dflash2.safetensors
python scripts/dflash2_forward_reference.py \
  --archive /path/to/specforge-source.tar.gz \
  --output-directory Tests/MLXLMTests/Resources
python scripts/prism_speculative_prompts.py \
  --model /path/to/target --output /tmp/dflash2-prompts.json \
  --include-sliding-window
```

The opt-in real-checkpoint test compares each generated token against ordinary greedy decoding, checks that drafts are proposed and accepted, and covers a prompt beyond the drafter's sliding window. Tokenization for this test uses Python Transformers; it is not an independent Swift tokenizer test.

```bash
TEST_RUNNER_PRISM_HADAMARD_MODEL=/path/to/target \
TEST_RUNNER_PRISM_DFLASH2_MODEL=/path/to/drafter \
TEST_RUNNER_PRISM_SPEC_PROMPTS=/tmp/dflash2-prompts.json \
xcodebuild test -scheme mlx-swift-lm-Package \
  -destination 'platform=macOS' -configuration Release \
  -skipPackagePluginValidation ENABLE_TESTABILITY=YES \
  -only-testing:MLXLMTests/DFlash2Tests
```
