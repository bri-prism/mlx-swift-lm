import Foundation
import MLX
import MLXNN

public enum PrismHadamardLoadError: Error {
    case invalidManifest
    case invalidModule(String)
}

struct PrismHadamardManifest: Decodable {
    struct Record: Decodable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }
    struct Quantization: Decodable {
        let bits: Int
        let group_size: Int
        let mode: String
    }
    let schema_version: Int
    let model_type: String
    let base_model_type: String?
    let quantization: Quantization
    let modules: [Record]

    static func load(from directory: URL) throws -> Self? {
        let url = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        let header = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard header?["model_type"] as? String == "prism_hadamard_qwen35" else {
            return nil
        }
        let manifest = try JSONDecoder().decode(Self.self, from: data)
        guard [1, 2].contains(manifest.schema_version),
            manifest.base_model_type == nil || manifest.base_model_type == "qwen3_5",
            manifest.quantization.bits == 2, manifest.quantization.group_size == 128,
            manifest.quantization.mode == "affine", !manifest.modules.isEmpty,
            Set(manifest.modules.map(\.path)).count == manifest.modules.count
        else { throw PrismHadamardLoadError.invalidManifest }
        return manifest
    }
}

private final class PrismPackedEmbedding: QuantizedEmbedding {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x).asType(.float16)
    }
}

/// Installs the explicitly listed packed modules after validating their original dimensions.
func installPrismHadamardModules(
    _ manifest: PrismHadamardManifest, model: Module, weights: inout [String: MLXArray]
) throws {
    let originals = Dictionary(uniqueKeysWithValues: model.namedModules())
    var replacements = [(String, Module)]()
    var consumedSigns = [String]()
    var paths = Set<String>()
    for record in manifest.modules {
        let path = originals[record.path] != nil ? record.path : "language_model." + record.path
        guard paths.insert(path).inserted, record.dtype == "float16", !record.path.isEmpty,
            let original = originals[path],
            let shape = (original as? Linear)?.weight.shape
                ?? (original as? Embedding)?.weight.shape,
            shape.count == 2, shape[0] > 0, shape[1] % 128 == 0,
            record.embedding == (original is Embedding),
            original is Linear || original is Embedding,
            let weight = weights[path + ".weight"],
            let scales = weights[path + ".scales"],
            let biases = weights[path + ".biases"],
            weight.dtype == .uint32, weight.shape == [shape[0], shape[1] / 16],
            scales.shape == [shape[0], shape[1] / 128], biases.shape == scales.shape,
            [DType.float16, .float32, .bfloat16].contains(scales.dtype),
            biases.dtype == scales.dtype,
            all(isFinite(scales)).item(Bool.self), all(isFinite(biases)).item(Bool.self)
        else { throw PrismHadamardLoadError.invalidModule(record.path) }

        let signs = weights[path + ".signs"]
        let transform: SignedBlockHadamard?
        if record.block == 0 {
            guard signs == nil else { throw PrismHadamardLoadError.invalidModule(record.path) }
            transform = nil
        } else {
            guard [512, 1024, 2048, 4096].contains(record.block),
                shape[1] % record.block == 0, let signs, signs.shape == [shape[1]],
                [DType.float16, .float32, .bfloat16].contains(signs.dtype)
            else { throw PrismHadamardLoadError.invalidModule(record.path) }
            transform = try SignedBlockHadamard(
                blockSize: record.block, signs: signs.asArray(Float.self))
            consumedSigns.append(path + ".signs")
        }
        let replacement: Module
        if record.embedding {
            if let transform {
                replacement = try HadamardQuantizedEmbedding(
                    weight: weight, scales: scales, biases: biases, groupSize: 128, bits: 2,
                    transform: transform, outputDType: .float16)
            } else {
                replacement = PrismPackedEmbedding(
                    weight: weight, scales: scales, biases: biases, groupSize: 128, bits: 2)
            }
        } else {
            let bias = weights[path + ".bias"]
            guard ((original as? Linear)?.bias == nil) == (bias == nil),
                bias == nil || bias!.shape == [shape[0]]
            else {
                throw PrismHadamardLoadError.invalidModule(record.path)
            }
            if let transform {
                replacement = try HadamardQuantizedLinear(
                    weight: weight, bias: bias, scales: scales, biases: biases,
                    groupSize: 128, bits: 2, transform: transform)
            } else {
                replacement = QuantizedLinear(
                    weight: weight, bias: bias, scales: scales, biases: biases,
                    groupSize: 128, bits: 2)
            }
        }
        replacements.append((path, replacement))
    }
    try model.update(modules: ModuleChildren.unflattened(replacements), verify: [.noUnusedKeys])
    for key in consumedSigns { weights.removeValue(forKey: key) }
}
