import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class DFlash2Tests: XCTestCase {
    func testRejectsUnsupportedConfigurationsAndMissingWeights() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "dflash2-forward", withExtension: "json"))
        let data = try Data(contentsOf: url)
        let source = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for key in [
            "conv_group_size", "conv_kernel_size", "target_layer_ids", "selector_top_k",
            "attention_mode",
        ] {
            var root = source
            var options = try XCTUnwrap(root["dflash_config"] as? [String: Any])
            switch key {
            case "target_layer_ids": options[key] = [0, 4]
            case "attention_mode": options[key] = "mla"
            case "conv_group_size": options[key] = 3
            default: options[key] = 0
            }
            root["dflash_config"] = options
            let c = try JSONDecoder().decode(
                DFlash2Configuration.self, from: JSONSerialization.data(withJSONObject: root))
            XCTAssertThrowsError(try DFlash2DraftModel(c), key)
        }
        let c = try JSONDecoder().decode(DFlash2Configuration.self, from: data)
        let model = try DFlash2DraftModel(c)
        XCTAssertThrowsError(try model.update(parameters: .unflattened([:]), verify: [.all]))
    }

    func testCompleteForwardAgainstPublicPyTorchSource() throws {
        let configURL = try XCTUnwrap(
            Bundle.module.url(forResource: "dflash2-forward", withExtension: "json"))
        let weightsURL = try XCTUnwrap(
            Bundle.module.url(forResource: "dflash2-forward", withExtension: "safetensors"))
        let c = try JSONDecoder().decode(
            DFlash2Configuration.self, from: Data(contentsOf: configURL))
        let model = try DFlash2DraftModel(c)
        let arrays = try loadArrays(url: weightsURL)
        try model.update(
            parameters: .unflattened(arrays.filter { !$0.key.hasPrefix("fixture.") }),
            verify: [.all])
        let hidden = model.hiddenStates(
            embeddings: arrays["fixture.embeddings"]!, targetHidden: arrays["fixture.context"]!,
            offset: 6)
        XCTAssertLessThan(abs(hidden - arrays["fixture.output"]!).max().item(Float.self), 2e-5)
    }

    func testConvolutionAndSelectorAgainstPublicPyTorchSource() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "dflash2", withExtension: "safetensors"))
        let arrays = try loadArrays(url: url)
        let conv = DFlash2GroupedConv(hiddenSize: 8, blockSize: 4, taps: 3, groupSize: 2)
        let selector = DFlash2Selector(hiddenSize: 8, vocabularySize: 13, rank: 3)
        func weights(_ prefix: String) -> [String: MLXArray] {
            Dictionary(
                uniqueKeysWithValues: arrays.filter { $0.key.hasPrefix(prefix) }
                    .map { (String($0.key.dropFirst(prefix.count)), $0.value) })
        }
        try conv.update(parameters: .unflattened(weights("conv.")), verify: [.all])
        try selector.update(parameters: .unflattened(weights("selector.")), verify: [.all])
        let (prepared, delta) = conv.prepare(arrays["input"]!)
        let finished = conv.convolve(arrays["sublayer"]!, delta: delta, side: 1)
        for (actual, name) in [(prepared, "prepared"), (delta, "delta"), (finished, "finished")] {
            XCTAssertLessThan(abs(actual - arrays[name]!).max().item(Float.self), 2e-6, name)
        }
        let ids = arrays["ids"]!.asType(.int32)
        let hidden = arrays["hidden"]!
        let unary = arrays["unary"]!
        var previous = arrays["anchor"]!.asType(.int32)
        var path = [MLXArray]()
        for i in 0 ..< 3 {
            let candidates = ids[0..., i, 0...]
            let scores = selector.scores(
                ids: candidates, unary: unary[0..., i, 0...],
                hidden: hidden[0..., i, 0...], previous: previous)
            if i == 0 {
                XCTAssertLessThan(abs(scores - arrays["scores"]!).max().item(Float.self), 2e-6)
            }
            previous = takeAlong(
                candidates, argMax(scores, axis: -1).expandedDimensions(axis: -1), axis: -1
            ).flattened()
            path.append(previous)
        }
        XCTAssertEqual(stacked(path, axis: 1).asArray(Int.self), arrays["path"]!.asArray(Int.self))
    }

    func testPublicCheckpointGreedyParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["PRISM_HADAMARD_MODEL"], let draftPath = env["PRISM_DFLASH2_MODEL"],
            let promptsPath = env["PRISM_SPEC_PROMPTS"]
        else { throw XCTSkip("Set the target, DFlash2 checkpoint and prompt fixture paths") }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let target = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        try await loadWeights(modelDirectory: directory, model: target)
        let drafter = try DFlash2DraftModel.load(from: URL(fileURLWithPath: draftPath))
        struct Prompt: Decodable {
            let name: String
            let input_ids: [Int]
        }
        let prompts = try JSONDecoder().decode(
            [Prompt].self,
            from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
        for prompt in prompts {
            var baseline = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt.input_ids)),
                model: target, parameters: .init(maxTokens: 64, temperature: 0))
            var expected = [Int]()
            while let token = baseline.next() {
                expected.append(token)
                if token == 248044 || token == 248046 { break }
            }
            var iterator = try DFlash2TokenIterator(
                input: MLXArray(prompt.input_ids), target: target,
                drafter: drafter, maxTokens: 64)
            var actual = [Int]()
            while let token = iterator.next() {
                actual.append(token)
                if token == 248044 || token == 248046 { break }
            }
            print(
                "DFlash2 \(prompt.name): proposed=\(iterator.proposedCount) accepted=\(iterator.acceptedCount) replay=\(iterator.replayCount) tokens=\(actual)"
            )
            XCTAssertGreaterThan(iterator.proposedCount, 0)
            XCTAssertGreaterThan(iterator.acceptedCount, 0)
            XCTAssertEqual(actual, expected, prompt.name)
        }
    }
}
