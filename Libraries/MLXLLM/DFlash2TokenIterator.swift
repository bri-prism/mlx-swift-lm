import Foundation
import MLX
import MLXLMCommon

/// Greedy, text-only DFlash2 generation for Qwen3.5 targets, including packed Bonsai.
///
/// The iterator owns fresh caches. Verification uses isolated copies; rejection replays
/// only committed tokens, preserving recurrent state. It does not support resuming an
/// external cache or stochastic sampling. Use `generateTask` with this iterator to stream.
public struct DFlash2TokenIterator: TokenIteratorProtocol {
    public let maxTokens: Int?
    public private(set) var tokenCount = 0
    public private(set) var promptPrefillTime: TimeInterval = 0
    public private(set) var proposedCount = 0
    public private(set) var acceptedCount = 0
    public private(set) var replayCount = 0
    private var roundCount = 0
    private var draftCalls = 0
    private var verifiedCount = 0
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        .init(
            roundCount: roundCount, draftTokenCount: proposedCount,
            acceptedDraftTokenCount: acceptedCount, targetModelCallCount: roundCount + replayCount,
            draftModelCallCount: draftCalls, targetVerifiedTokenCount: verifiedCount,
            emittedTokenCount: tokenCount)
    }

    private let target: Qwen35TextModel
    private let drafter: DFlash2DraftModel
    private var cache: [KVCache]
    private var context: MLXArray
    private var offset: Int
    private var anchor: Int
    private var pending: [Int]
    private var pendingIndex = 0

    public init(
        input: MLXArray, target: any LanguageModel, drafter: DFlash2DraftModel,
        maxTokens: Int
    ) throws {
        let text: Qwen35TextModel
        if let model = target as? Qwen35Model {
            text = model.languageModel
        } else if let model = target as? Qwen35TextModel {
            text = model
        } else {
            throw DFlash2Error.unsupported("DFlash2 currently requires a Qwen3.5 text target")
        }
        let c = drafter.configuration
        guard maxTokens >= 0, input.size > 0,
            input.ndim == 1 || (input.ndim == 2 && input.dim(0) == 1),
            input.dtype == .int32 || input.dtype == .int64 || input.dtype == .uint32,
            text.configuration.hiddenSize == c.hidden_size,
            text.vocabularySize == c.vocab_size,
            text.configuration.hiddenLayers == c.num_target_layers
        else { throw DFlash2Error.unsupported("DFlash2 target geometry or input is incompatible") }
        guard input.min().item(Int.self) >= 0, input.max().item(Int.self) < c.vocab_size else {
            throw DFlash2Error.unsupported("Input contains an invalid token ID")
        }
        self.maxTokens = maxTokens
        self.target = text
        self.drafter = drafter
        cache = try text.newCache(parameters: nil)
        offset = input.size
        let start = Date.timeIntervalSinceReferenceDate
        let (logits, hidden) = text.dflashForward(
            input.reshaped([1, -1]), cache: cache, taps: c.dflash_config.target_layer_ids)
        anchor = argMax(logits[0, -1, 0...]).item(Int.self)
        context = hidden
        pending = [anchor]
        eval(context)
        promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
        trimContext()
    }

    private mutating func trimContext() {
        // Full-attention draft layers require all target context; sliding-only models are bounded.
        let c = drafter.configuration
        if c.layer_types.allSatisfy({ $0 == "sliding_attention" }),
            let window = c.sliding_window, context.dim(1) > window
        {
            context = context[0..., (-window)..., 0...]
        }
    }

    private mutating func round() {
        let budget = maxTokens! - tokenCount
        let count = Swift.min(drafter.configuration.dflash_config.block_size - 1, budget - 1)
        let draft: [Int]
        if count > 0 {
            draft = Array(
                drafter.propose(target: target, anchor: anchor, context: context, offset: offset)
                    .prefix(count))
        } else {
            draft = []
        }
        let tokens = [anchor] + draft
        let verificationCache = cache.map { $0.copy() }
        let (logits, hidden) = target.dflashForward(
            MLXArray(tokens).reshaped([1, -1]), cache: verificationCache,
            taps: drafter.configuration.dflash_config.target_layer_ids)
        let predictions = argMax(logits[0], axis: -1).asArray(Int.self)
        var accepted = 0
        while accepted < draft.count && draft[accepted] == predictions[accepted] { accepted += 1 }
        let committed = accepted + 1
        if accepted == draft.count {
            cache = verificationCache
            context = concatenated([context, hidden], axis: 1)
        } else {
            let (_, replayedHidden) = target.dflashForward(
                MLXArray(Array(tokens.prefix(committed))).reshaped([1, -1]), cache: cache,
                taps: drafter.configuration.dflash_config.target_layer_ids)
            context = concatenated([context, replayedHidden], axis: 1)
            replayCount += 1
        }
        offset += committed
        trimContext()
        eval(context)
        anchor = predictions[accepted]
        pending = Array(draft.prefix(accepted)) + [anchor]
        pendingIndex = 0
        proposedCount += draft.count
        acceptedCount += accepted
        roundCount += 1
        draftCalls += draft.isEmpty ? 0 : 1
        verifiedCount += tokens.count
    }

    public mutating func next() -> Int? {
        guard tokenCount < maxTokens! else { return nil }
        if pendingIndex == pending.count { round() }
        let token = pending[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    public mutating func discardGeneratedToken() {
        tokenCount = Swift.max(0, tokenCount - 1)
    }
}
