import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class PrismSpeculativeTests: XCTestCase {
    func testStandaloneHeadRejectsDifferentTargetGeometry() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(
            """
            {"hidden_size":17,"intermediate_size":32,"num_attention_heads":2,
             "num_key_value_heads":1,"head_dim":8,"num_mtp_layers":1}
            """.utf8
        ).write(to: directory.appendingPathComponent("mtp_config.json"))
        let target = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"hidden_size":16,"intermediate_size":32,"num_attention_heads":2,
                 "num_key_value_heads":1,"head_dim":8}
                """.utf8))
        XCTAssertThrowsError(
            try Qwen35MTPDraftModel.loadHead(from: directory, targetConfiguration: target)
        ) { error in
            guard case DecodingError.dataCorrupted(let context) = error else {
                return XCTFail("Expected geometry rejection, got \(error)")
            }
            XCTAssertEqual(context.debugDescription, "MTP head and target geometry differ")
        }
    }

    func testPublicMTPGreedyParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let targetPath = environment["PRISM_HADAMARD_MODEL"],
            let draftPath = environment["PRISM_MTP_MODEL"],
            let promptsPath = environment["PRISM_SPEC_PROMPTS"]
        else { throw XCTSkip("Set target, public MTP head, and tokenized prompts paths") }
        let directory = URL(fileURLWithPath: targetPath)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let target = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        try await loadWeights(modelDirectory: directory, model: target)

        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let config = try XCTUnwrap(root["text_config"] as? [String: Any])
        let drafter = try Qwen35MTPDraftModel.loadHead(
            from: URL(fileURLWithPath: draftPath),
            targetConfiguration: JSONDecoder().decode(
                Qwen35TextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: config)))

        struct Prompt: Decodable {
            let name: String
            let input_ids: [Int]
        }
        let prompts = try JSONDecoder().decode(
            [Prompt].self, from: Data(contentsOf: URL(fileURLWithPath: promptsPath)))
        for prompt in prompts {
            let input = LMInput(tokens: MLXArray(prompt.input_ids))
            let parameters = GenerateParameters(maxTokens: 64, temperature: 0)
            var baseline = try TokenIterator(input: input, model: target, parameters: parameters)
            var expected = [Int]()
            while let token = baseline.next() {
                expected.append(token)
                if token == 248044 || token == 248046 { break }
            }
            var speculative = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: drafter,
                parameters: parameters, blockSize: 2)
            var actual = [Int]()
            while let token = speculative.next() {
                actual.append(token)
                if token == 248044 || token == 248046 { break }
            }
            print(
                "MTP \(prompt.name): proposed=\(speculative.proposedCount) accepted=\(speculative.acceptedCount) fallback=\(String(describing: speculative.passthroughReason)) tokens=\(actual)"
            )
            XCTAssertGreaterThan(speculative.proposedCount, 0)
            XCTAssertNil(speculative.passthroughReason)
            XCTAssertEqual(actual, expected, prompt.name)
        }
    }
}
