import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLMCommon

private final class PrismTestModules: Module {
    @ModuleInfo var projection: Linear
    @ModuleInfo var embedding: Embedding

    override init() {
        _projection.wrappedValue = Linear(512, 8, bias: false)
        _embedding.wrappedValue = Embedding(embeddingCount: 8, dimensions: 512)
    }
}

final class PrismHadamardLoaderTests: XCTestCase {
    private func fixture() throws -> (PrismTestModules, PrismHadamardManifest, [String: MLXArray]) {
        let model = PrismTestModules()
        var weights = [String: MLXArray]()
        for name in ["projection", "embedding"] {
            let w = MLXRandom.normal([8, 512]).asType(.float16)
            let (q, s, b) = quantized(w, groupSize: 128, bits: 2)
            weights[name + ".weight"] = q
            weights[name + ".scales"] = s
            weights[name + ".biases"] = b
            weights[name + ".signs"] = MLXArray.ones([512])
        }
        let data = Data(
            """
            {"schema_version":2,"model_type":"prism_hadamard_qwen35",
             "base_model_type":"qwen3_5","quantization":{"bits":2,"group_size":128,"mode":"affine"},
             "modules":[{"path":"projection","block":512,"embedding":false,"dtype":"float16"},
                        {"path":"embedding","block":512,"embedding":true,"dtype":"float16"}]}
            """.utf8)
        return (model, try JSONDecoder().decode(PrismHadamardManifest.self, from: data), weights)
    }

    func testPackedLayersAndStrictUpdate() throws {
        let (model, manifest, source) = try fixture()
        var weights = source
        try installPrismHadamardModules(manifest, model: model, weights: &weights)
        try model.update(parameters: .unflattened(weights), verify: [.all])
        XCTAssertTrue(model.projection is HadamardQuantizedLinear)
        XCTAssertTrue(model.embedding is HadamardQuantizedEmbedding)
        XCTAssertNil(weights["projection.signs"])
        let x = MLXRandom.normal([2, 512]).asType(.float16)
        let transformed = hadamardTransform(x.asType(.float32)).asType(.float16)
        let expected = quantizedMM(
            transformed, source["projection.weight"]!, scales: source["projection.scales"]!,
            biases: source["projection.biases"]!, groupSize: 128, bits: 2)
        XCTAssertTrue(all(model.projection(x) .== expected).item(Bool.self))
        XCTAssertEqual(model.embedding(MLXArray([0, 1])).dtype, .float16)
    }

    func testRejectsInvalidPackedShapeSignsAndDType() throws {
        for corruption in 0 ..< 5 {
            let (model, manifest, source) = try fixture()
            var weights = source
            switch corruption {
            case 0: weights["projection.weight"] = MLXArray.zeros([8, 16], dtype: .uint32)
            case 1: weights["projection.signs"] = MLXArray.zeros([512])
            case 2: weights["projection.signs"] = MLXArray.ones([256])
            case 3: weights["projection.scales"] = MLXArray.zeros([8, 4], dtype: .int32)
            default:
                weights["projection.scales"] = MLXArray.full([8, 4], values: MLXArray(Float.nan))
            }
            XCTAssertThrowsError(
                try installPrismHadamardModules(manifest, model: model, weights: &weights))
            XCTAssertFalse(model.projection is HadamardQuantizedLinear)
        }
    }

    func testRejectsInvalidManifest() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let valid: [String: Any] = [
            "schema_version": 2, "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": [
                ["path": "projection", "block": 512, "embedding": false, "dtype": "float16"]
            ],
        ]
        for key in ["schema_version", "base_model_type", "quantization", "modules"] {
            var invalid = valid
            switch key {
            case "schema_version": invalid[key] = 3
            case "base_model_type": invalid[key] = "qwen2"
            case "quantization": invalid[key] = ["bits": 4, "group_size": 128, "mode": "affine"]
            default: invalid[key] = [] as [String]
            }
            try JSONSerialization.data(withJSONObject: invalid).write(
                to: directory.appendingPathComponent("config.json"))
            XCTAssertThrowsError(try PrismHadamardManifest.load(from: directory))
        }
    }

    func testLocalHadamardCheckpoint() async throws {
        guard let path = ProcessInfo.processInfo.environment["PRISM_HADAMARD_MODEL"] else {
            throw XCTSkip("Set PRISM_HADAMARD_MODEL to a local packed checkpoint")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        try await loadWeights(modelDirectory: directory, model: model)
        guard let referencePath = ProcessInfo.processInfo.environment["PRISM_HADAMARD_REFERENCE"]
        else {
            let output = model(MLXArray([1, 2, 3]).reshaped([1, 3]), cache: nil)
            eval(output)
            XCTAssertTrue(all(isFinite(output)).item(Bool.self))
            return
        }
        struct Reference: Decodable {
            let input_ids: [Int]
            let output_ids: [Int]
        }
        let reference = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: URL(fileURLWithPath: referencePath + ".json")))
        let logits = try loadArrays(url: URL(fileURLWithPath: referencePath + ".safetensors"))
        let cache = try model.newCache(parameters: nil)
        var input = MLXArray(reference.input_ids).reshaped([1, -1])
        for (step, expectedToken) in reference.output_ids.enumerated() {
            let output = model(input, cache: cache)[0..., -1, 0...].asType(.float32)
            eval(output)
            XCTAssertTrue(all(isFinite(output)).item(Bool.self))
            XCTAssertEqual(argMax(output).item(Int.self), expectedToken)
            let expected = try XCTUnwrap(logits["logits_\(step)"])
            let logP = expected - logSumExp(expected, axis: -1, keepDims: true)
            let logQ = output - logSumExp(output, axis: -1, keepDims: true)
            let divergence = sum(exp(logP) * (logP - logQ)).item(Float.self)
            print(
                "Prism step \(step): KL=\(divergence), max error=\(abs(output - expected).max().item(Float.self))"
            )
            XCTAssertLessThan(divergence, 0.001)
            input = MLXArray([expectedToken]).reshaped([1, 1])
        }
    }
    func testLocalVisionCheckpoint() async throws {
        guard let path = ProcessInfo.processInfo.environment["PRISM_HADAMARD_MODEL"],
            let referencePath = ProcessInfo.processInfo.environment["PRISM_HADAMARD_REFERENCE"]
        else { throw XCTSkip("Set local checkpoint and reference paths") }
        let directory = URL(fileURLWithPath: path)
        struct TestLoader: TokenizerLoader {
            func load(from directory: URL) async throws -> any Tokenizer { TestTokenizer() }
        }
        let context = try await VLMModelFactory.shared.load(from: directory, using: TestLoader())
        let model = context.model
        struct Reference: Decodable {
            let input_ids: [Int]
            let output_ids: [Int]
            let grid: [[Int]]
        }
        for color in ["red", "blue"] {
            let base = referencePath + "-vision-" + color
            let reference = try JSONDecoder().decode(
                Reference.self, from: Data(contentsOf: URL(fileURLWithPath: base + ".json")))
            let arrays = try loadArrays(url: URL(fileURLWithPath: base + ".safetensors"))
            let image = LMInput.ProcessedImage(
                pixels: try XCTUnwrap(arrays["pixels"]),
                frames: reference.grid.map { THW($0[0], $0[1], $0[2]) })
            let cache = try model.newCache(parameters: nil)
            let input = LMInput(
                text: .init(tokens: MLXArray(reference.input_ids).reshaped([1, -1])), image: image)
            guard
                case .logits(var result) = try model.prepare(
                    input, cache: cache, state: nil, prefill: .init())
            else { return XCTFail("Expected VLM prefill logits") }
            for (step, token) in reference.output_ids.enumerated() {
                let logits = result.logits[0..., -1, 0...].asType(.float32)
                XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
                XCTAssertEqual(argMax(logits).item(Int.self), token)
                let expected = try XCTUnwrap(arrays["logits_\(step)"])
                let logP = expected - logSumExp(expected, axis: -1, keepDims: true)
                let logQ = logits - logSumExp(logits, axis: -1, keepDims: true)
                let divergence = sum(exp(logP) * (logP - logQ)).item(Float.self)
                print("Prism \(color) step \(step): KL=\(divergence)")
                XCTAssertLessThan(divergence, 0.001)
                if step + 1 < reference.output_ids.count {
                    result = model(
                        .init(tokens: MLXArray([token]).reshaped([1, 1])),
                        cache: cache, state: result.state)
                }
            }
        }
    }

}
