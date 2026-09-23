// DFlash2 architecture adapted from SpecForge (MIT, Copyright 2025 sgl-project).
// See ThirdPartyLicenses/SpecForge.txt for the license and source attribution.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum DFlash2Error: Error, LocalizedError {
    case unsupported(String)
    public var errorDescription: String? {
        switch self {
        case .unsupported(let message): return message
        }
    }
}

/// Checkpoint configuration for the Qwen3 GQA DFlash2 architecture.
public struct DFlash2Configuration: Decodable, Sendable {
    struct Options: Decodable, Sendable {
        let block_size: Int
        let conv_group_size: Int
        let conv_kernel_size: Int
        let mask_token_id: Int
        let selector_rank: Int
        let selector_top_k: Int
        let target_layer_ids: [Int]
        let output_multiplier: Float?
        let final_logit_softcapping: Float?
        let attention_mode: String?
        let shift_label: Bool?
        let pure_draft_prefix_len: Int?
        let projector_type: String?
    }
    struct Rotary: Decodable, Sendable {
        let rope_theta: Float
        let rope_type: String
    }
    let architectures: [String]
    let model_type: String
    let hidden_size: Int
    let intermediate_size: Int
    let num_attention_heads: Int
    let num_key_value_heads: Int
    let head_dim: Int
    let num_hidden_layers: Int
    let num_target_layers: Int
    let vocab_size: Int
    let rms_norm_eps: Float
    let attention_bias: Bool
    let hidden_act: String
    let is_causal: Bool
    let layer_types: [String]
    let sliding_window: Int?
    let rope_parameters: Rotary
    let dflash_config: Options

    func validate() throws {
        let d = dflash_config
        guard architectures == ["DFlash2DraftModel"], model_type == "qwen3",
            hidden_act == "silu", !is_causal, !attention_bias,
            hidden_size > 0, intermediate_size > 0, num_hidden_layers > 0,
            num_attention_heads > 0, num_key_value_heads > 0,
            num_attention_heads % num_key_value_heads == 0,
            head_dim > 0, head_dim % 2 == 0, vocab_size > 0,
            rms_norm_eps.isFinite, rms_norm_eps > 0,
            rope_parameters.rope_type == "default",
            rope_parameters.rope_theta.isFinite, rope_parameters.rope_theta > 0,
            layer_types.count == num_hidden_layers,
            layer_types.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" }),
            !layer_types.contains("sliding_attention") || (sliding_window ?? 0) > 0,
            d.block_size >= 2, d.conv_group_size > 0,
            hidden_size % d.conv_group_size == 0,
            d.conv_kernel_size > 0, d.conv_kernel_size <= d.block_size,
            d.selector_rank > 0, d.selector_top_k > 0, d.selector_top_k <= vocab_size,
            (0 ..< vocab_size).contains(d.mask_token_id),
            !d.target_layer_ids.isEmpty,
            Set(d.target_layer_ids).count == d.target_layer_ids.count,
            d.target_layer_ids.allSatisfy({ (0 ..< num_target_layers).contains($0) }),
            (d.output_multiplier ?? 1).isFinite,
            (d.final_logit_softcapping ?? 1).isFinite,
            (d.final_logit_softcapping ?? 1) > 0,
            (d.attention_mode ?? "gqa") == "gqa",
            !(d.shift_label ?? false), (d.pure_draft_prefix_len ?? 0) == 0,
            d.projector_type == nil
        else { throw DFlash2Error.unsupported("Unsupported or malformed DFlash2 configuration") }
    }
}

final class DFlash2GroupedConv: Module {
    let blockSize: Int
    let taps: Int
    let groupSize: Int
    let groups: Int
    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var projection: Linear

    init(hiddenSize: Int, blockSize: Int, taps: Int, groupSize: Int) {
        self.blockSize = blockSize
        self.taps = taps
        self.groupSize = groupSize
        self.groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, taps, hiddenSize])
        _projection.wrappedValue = Linear(hiddenSize, 2 * taps * groups, bias: false)
    }

    func convolve(_ x: MLXArray, delta: MLXArray, side: Int) -> MLXArray {
        let b = x.dim(0)
        let length = x.dim(1)
        let h = x.dim(2)
        precondition(length % blockSize == 0)
        let blocks = x.reshaped([b, length / blockSize, blockSize, groups, groupSize])
        let dynamic = delta.reshaped([b, length / blockSize, blockSize, taps, groups])
        var result = MLXArray.zeros(like: blocks)
        for tap in 0 ..< taps {
            let coefficient =
                baseKernel[side, tap, 0...].reshaped([1, 1, 1, groups, groupSize])
                + dynamic[0..., 0..., 0..., tap, 0...].expandedDimensions(axis: -1)
            let shifted: MLXArray
            if tap == 0 {
                shifted = blocks
            } else {
                shifted = concatenated(
                    [
                        MLXArray.zeros(
                            [b, length / blockSize, tap, groups, groupSize], dtype: x.dtype),
                        blocks[0..., 0..., ..<(blockSize - tap), 0..., 0...],
                    ], axis: 2)
            }
            result = result + coefficient * shifted
        }
        return result.reshaped(x.shape)
    }

    func prepare(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let delta = projection(x).reshaped([x.dim(0), x.dim(1), 2, taps, groups])
        return (
            convolve(x, delta: delta[0..., 0..., 0, 0..., 0...], side: 0),
            delta[0..., 0..., 1, 0..., 0...]
        )
    }
}

final class DFlash2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let rope: RoPE
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: DFlash2Configuration) {
        heads = c.num_attention_heads
        kvHeads = c.num_key_value_heads
        headDim = c.head_dim
        rope = RoPE(dimensions: headDim, traditional: false, base: c.rope_parameters.rope_theta)
        _q.wrappedValue = Linear(c.hidden_size, heads * headDim, bias: false)
        _k.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _v.wrappedValue = Linear(c.hidden_size, kvHeads * headDim, bias: false)
        _o.wrappedValue = Linear(heads * headDim, c.hidden_size, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rms_norm_eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.rms_norm_eps)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, offset: Int, mask: MLXArray?) -> MLXArray
    {
        let b = x.dim(0)
        let length = x.dim(1)
        let contextLength = context.dim(1)
        let queries = qNorm(q(x).reshaped([b, length, heads, headDim])).transposed(0, 2, 1, 3)
        let keys = kNorm(
            concatenated([k(context), k(x)], axis: 1)
                .reshaped([b, contextLength + length, kvHeads, headDim])
        ).transposed(0, 2, 1, 3)
        let values = concatenated([v(context), v(x)], axis: 1)
            .reshaped([b, contextLength + length, kvHeads, headDim]).transposed(0, 2, 1, 3)
        let result = MLXFast.scaledDotProductAttention(
            queries: rope(queries, offset: offset),
            keys: rope(keys, offset: offset - contextLength), values: values,
            scale: 1 / sqrt(Float(headDim)), mask: mask)
        return o(result.transposed(0, 2, 1, 3).reshaped([b, length, heads * headDim]))
    }
}

final class DFlash2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: DFlash2Configuration) {
        _gate.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _up.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _down.wrappedValue = Linear(c.intermediate_size, c.hidden_size, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class DFlash2Layer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2GroupedConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2GroupedConv
    init(_ c: DFlash2Configuration) {
        _attention.wrappedValue = DFlash2Attention(c)
        _mlp.wrappedValue = DFlash2MLP(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.rms_norm_eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.rms_norm_eps)
        let d = c.dflash_config
        _attentionConv.wrappedValue = DFlash2GroupedConv(
            hiddenSize: c.hidden_size, blockSize: d.block_size,
            taps: d.conv_kernel_size, groupSize: d.conv_group_size)
        _mlpConv.wrappedValue = DFlash2GroupedConv(
            hiddenSize: c.hidden_size, blockSize: d.block_size,
            taps: d.conv_kernel_size, groupSize: d.conv_group_size)
    }
    func callAsFunction(_ x: MLXArray, context: MLXArray, offset: Int, mask: MLXArray?) -> MLXArray
    {
        let (input, attentionDelta) = attentionConv.prepare(inputNorm(x))
        let h =
            x
            + attentionConv.convolve(
                attention(input, context: context, offset: offset, mask: mask),
                delta: attentionDelta, side: 1)
        let (mlpInput, mlpDelta) = mlpConv.prepare(postNorm(h))
        return h + mlpConv.convolve(mlp(mlpInput), delta: mlpDelta, side: 1)
    }
}

final class DFlash2Selector: Module {
    @ParameterInfo(key: "predecessor_codebook") var predecessor: MLXArray
    @ParameterInfo(key: "successor_codebook") var successor: MLXArray
    @ModuleInfo(key: "hidden_projection") var projection: Linear
    init(hiddenSize: Int, vocabularySize: Int, rank: Int) {
        _predecessor.wrappedValue = MLXArray.zeros([vocabularySize, rank])
        _successor.wrappedValue = MLXArray.zeros([vocabularySize, rank])
        _projection.wrappedValue = Linear(hiddenSize, rank, bias: false)
    }
    func scores(ids: MLXArray, unary: MLXArray, hidden: MLXArray, previous: MLXArray) -> MLXArray {
        let context = predecessor[previous] * projection(hidden)
        return unary + sum(context.expandedDimensions(axis: -2) * successor[ids], axis: -1)
    }
}

/// The public DFlash2 GQA drafter. Uses the target's embedding and output head unchanged.
public final class DFlash2DraftModel: Module {
    public let configuration: DFlash2Configuration
    @ModuleInfo var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var layers: [DFlash2Layer]
    @ModuleInfo(key: "candidate_selector") var selector: DFlash2Selector

    public init(_ c: DFlash2Configuration) throws {
        try c.validate()
        configuration = c
        _fc.wrappedValue = Linear(
            c.hidden_size * c.dflash_config.target_layer_ids.count, c.hidden_size, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.rms_norm_eps)
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.rms_norm_eps)
        _layers.wrappedValue = (0 ..< c.num_hidden_layers).map { _ in DFlash2Layer(c) }
        _selector.wrappedValue = DFlash2Selector(
            hiddenSize: c.hidden_size,
            vocabularySize: c.vocab_size, rank: c.dflash_config.selector_rank)
    }

    public static func load(from directory: URL) throws -> DFlash2DraftModel {
        let c = try JSONDecoder().decode(
            DFlash2Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = try DFlash2DraftModel(c)
        let weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        try model.update(parameters: .unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    func hiddenStates(embeddings: MLXArray, targetHidden: MLXArray, offset: Int) -> MLXArray {
        let context = hiddenNorm(fc(targetHidden.asType(fc.weight.dtype)))
        var hidden = embeddings.asType(fc.weight.dtype)
        for (i, layer) in layers.enumerated() {
            let mask: MLXArray?
            if configuration.layer_types[i] == "sliding_attention" {
                let positions = MLXArray((offset - context.dim(1)) ..< (offset + hidden.dim(1)))
                let queries = MLXArray(offset ..< (offset + hidden.dim(1)))
                let lowerBound =
                    queries.expandedDimensions(axis: -1) - (configuration.sliding_window! - 1)
                // Every proposal row sees the whole draft block, including future mask slots.
                mask = ((positions .>= lowerBound) .|| (positions .>= offset))
                    .reshaped([1, 1, hidden.dim(1), positions.size])
            } else {
                mask = nil
            }
            hidden = layer(hidden, context: context, offset: offset, mask: mask)
        }
        return norm(hidden)
    }

    func propose(target: Qwen35TextModel, anchor: Int, context: MLXArray, offset: Int) -> [Int] {
        let d = configuration.dflash_config
        let tokens = MLXArray([anchor] + Array(repeating: d.mask_token_id, count: d.block_size - 1))
            .reshaped([1, d.block_size])
        let hidden = hiddenStates(
            embeddings: target.model.embedTokens(tokens), targetHidden: context, offset: offset)
        let proposalHidden = hidden[0..., 1..., 0...]
        var logits = target.dflashHead(proposalHidden).asType(.float32) * (d.output_multiplier ?? 1)
        if let cap = d.final_logit_softcapping { logits = tanh(logits / cap) * cap }
        let candidates = argSort(-logits, axis: -1)[0..., 0..., ..<d.selector_top_k]
        let unary = takeAlong(logits, candidates, axis: -1)
        var previous = MLXArray([anchor])
        var result = [Int]()
        for i in 0 ..< d.block_size - 1 {
            let ids = candidates[0..., i, 0...]
            let scores = selector.scores(
                ids: ids, unary: unary[0..., i, 0...],
                hidden: proposalHidden[0..., i, 0...], previous: previous)
            previous = takeAlong(
                ids, argMax(scores, axis: -1).expandedDimensions(axis: -1), axis: -1
            ).flattened()
            result.append(previous.item(Int.self))
        }
        return result
    }
}

extension Qwen35TextModel {
    func dflashHead(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }
    func dflashForward(_ tokens: MLXArray, cache: [KVCache], taps: [Int]) -> (MLXArray, MLXArray) {
        var captured = [Int: MLXArray]()
        let wanted = Set(taps)
        let hidden = model.forward(tokens, cache: cache, applyFinalNorm: true) { index, value in
            if wanted.contains(index) { captured[index] = value }
        }
        return (dflashHead(hidden), concatenated(taps.map { captured[$0]! }, axis: -1))
    }
}
