// Conformance test: Manual OTel instrumentation for AWS Bedrock Java.
//
// Exercises: converse (Bedrock Converse API)
// against a mock Bedrock server, with manual OTel span creation around raw SDK calls.

package com.example.bedrocktest;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ConversationRole;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseRequest;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseResponse;
import software.amazon.awssdk.services.bedrockruntime.model.Message;

import java.net.URI;
import java.util.List;

public class AwsBedrockPrototypeTest {

    private static final Tracer tracer = GlobalOpenTelemetry.getTracer("gen_ai.prototype");
    private static URI mockEndpoint;

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");
        mockEndpoint = new URI(mockBaseUrl);

        System.out.println("=== Prototype: AWS Bedrock Java Conformance Test ===");

        // Create raw client - NO instrumentation interceptor
        BedrockRuntimeClient client = BedrockRuntimeClient.builder()
            .endpointOverride(mockEndpoint)
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .build();

        // Run scenarios
        runConverse(client);

        client.close();
        System.out.println("Done.");
    }

    static void runConverse(BedrockRuntimeClient client) {
        System.out.println("  [converse] Bedrock Converse API");
        String modelId = "anthropic.claude-3-haiku-20240307-v1:0";
        Span span = tracer.spanBuilder("chat " + modelId).startSpan();
        try {
            span.setAttribute("gen_ai.operation.name", "chat");
            span.setAttribute("gen_ai.provider.name", "aws.bedrock");
            span.setAttribute("gen_ai.request.model", modelId);
            if (mockEndpoint.getHost() != null) {
                span.setAttribute("server.address", mockEndpoint.getHost());
            }
            if (mockEndpoint.getPort() != -1) {
                span.setAttribute("server.port", (long) mockEndpoint.getPort());
            }

            ConverseResponse response = client.converse(ConverseRequest.builder()
                    .modelId(modelId)
                    .messages(Message.builder()
                            .role(ConversationRole.USER)
                            .content(ContentBlock.fromText("Say hello."))
                            .build())
                    .build());

            span.setAttribute(AttributeKey.stringArrayKey("gen_ai.response.finish_reasons"),
                    List.of(response.stopReason().toString()));
            span.setAttribute("gen_ai.usage.input_tokens", (long) response.usage().inputTokens());
            span.setAttribute("gen_ai.usage.output_tokens", (long) response.usage().outputTokens());

            String text = response.output().message().content().get(0).text();
            System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
        } finally {
            span.end();
        }
    }
}
