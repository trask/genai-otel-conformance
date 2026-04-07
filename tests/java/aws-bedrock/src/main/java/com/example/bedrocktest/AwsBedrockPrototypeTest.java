// Conformance test: Manual OTel instrumentation for AWS Bedrock Java.
//
// Exercises: converse (Bedrock Converse API)
// against a mock Bedrock server, with manual OTel span creation around raw SDK calls.

package com.example.bedrocktest;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.KeyValue;
import io.opentelemetry.api.common.Value;
import io.opentelemetry.api.logs.Logger;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.instrumentation.awssdk.v2_2.AwsSdkTelemetry;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ConversationRole;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseRequest;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseResponse;
import software.amazon.awssdk.services.bedrockruntime.model.Message;
import software.amazon.awssdk.services.bedrockruntime.model.Tool;
import software.amazon.awssdk.services.bedrockruntime.model.ToolConfiguration;
import software.amazon.awssdk.services.bedrockruntime.model.ToolInputSchema;
import software.amazon.awssdk.services.bedrockruntime.model.ToolSpecification;

import software.amazon.awssdk.core.document.Document;

import java.net.URI;
import java.util.List;

import static io.opentelemetry.api.common.AttributeKey.longKey;
import static io.opentelemetry.api.common.AttributeKey.stringArrayKey;
import static io.opentelemetry.api.common.AttributeKey.stringKey;
import static io.opentelemetry.api.common.AttributeKey.valueKey;

public class AwsBedrockPrototypeTest {

    private static final Tracer tracer = GlobalOpenTelemetry.getTracer("gen_ai.prototype");
    private static final Logger eventLogger =
            GlobalOpenTelemetry.get().getLogsBridge().get("gen_ai.prototype");
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
        runConverseToolCall(client);

        client.close();

        // Run memory scenarios
        AwsSdkTelemetry telemetry = AwsSdkTelemetry.builder(GlobalOpenTelemetry.get()).build();
        AwsBedrockOtelContribTest.runMemoryOperations(mockBaseUrl, telemetry);

        System.out.println("Done.");
    }

    static void runConverse(BedrockRuntimeClient client) {
        System.out.println("  [converse] Bedrock Converse API");
        String modelId = "anthropic.claude-3-haiku-20240307-v1:0";
        Span span = tracer.spanBuilder("chat " + modelId).startSpan();
        try {
            try (var scope = span.makeCurrent()) {
                span.setAttribute(stringKey("gen_ai.operation.name"), "chat");
                span.setAttribute(stringKey("gen_ai.provider.name"), "aws.bedrock");
                span.setAttribute(stringKey("gen_ai.request.model"), modelId);
                if (mockEndpoint.getHost() != null) {
                    span.setAttribute(stringKey("server.address"), mockEndpoint.getHost());
                }
                if (mockEndpoint.getPort() != -1) {
                    span.setAttribute(longKey("server.port"), (long) mockEndpoint.getPort());
                }

                String userMessage = "Say hello.";
                ConverseResponse response = client.converse(ConverseRequest.builder()
                        .modelId(modelId)
                        .messages(Message.builder()
                                .role(ConversationRole.USER)
                                .content(ContentBlock.fromText(userMessage))
                                .build())
                        .build());

                span.setAttribute(stringArrayKey("gen_ai.response.finish_reasons"),
                        List.of(response.stopReason().toString()));
                span.setAttribute(longKey("gen_ai.usage.input_tokens"), (long) response.usage().inputTokens());
                span.setAttribute(longKey("gen_ai.usage.output_tokens"), (long) response.usage().outputTokens());

                String text = response.output().message().content().get(0).text();

                // Emit inference operation details event
                Value<?> inputMessages = Value.of(
                        Value.of(
                                KeyValue.of("role", Value.of("user")),
                                KeyValue.of("parts", Value.of(
                                        Value.of(
                                                KeyValue.of("type", Value.of("text")),
                                                KeyValue.of("content", Value.of(userMessage))
                                        )
                                ))
                        )
                );
                Value<?> outputMessages = Value.of(
                        Value.of(
                                KeyValue.of("role", Value.of("assistant")),
                                KeyValue.of("parts", Value.of(
                                        Value.of(
                                                KeyValue.of("type", Value.of("text")),
                                                KeyValue.of("content", Value.of(text))
                                        )
                                )),
                                KeyValue.of("finish_reason", Value.of(response.stopReason().toString()))
                        )
                );
                eventLogger.logRecordBuilder()
                        .setEventName("gen_ai.client.inference.operation.details")
                        .setAttribute(stringKey("gen_ai.operation.name"), "chat")
                        .setAttribute(stringKey("gen_ai.request.model"), modelId)
                        .setAttribute(stringArrayKey("gen_ai.response.finish_reasons"),
                                List.of(response.stopReason().toString()))
                        .setAttribute(longKey("gen_ai.usage.input_tokens"), (long) response.usage().inputTokens())
                        .setAttribute(longKey("gen_ai.usage.output_tokens"), (long) response.usage().outputTokens())
                        .setAttribute(valueKey("gen_ai.input.messages"), inputMessages)
                        .setAttribute(valueKey("gen_ai.output.messages"), outputMessages)
                        .emit();

                System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
            }
        } finally {
            span.end();
        }
    }

    static void runConverseToolCall(BedrockRuntimeClient client) {
        System.out.println("  [chat_tool_call] Bedrock Converse API with tool calling");
        String modelId = "anthropic.claude-3-haiku-20240307-v1:0";
        ToolSpecification toolSpec = ToolSpecification.builder()
                .name("get_weather")
                .description("Get the current weather")
                .inputSchema(ToolInputSchema.builder()
                        .json(Document.mapBuilder()
                                .putString("type", "object")
                                .putDocument("properties", Document.mapBuilder()
                                        .putDocument("location", Document.mapBuilder()
                                                .putString("type", "string")
                                                .putString("description", "City name")
                                                .build())
                                        .build())
                                .putList("required", List.of(Document.fromString("location")))
                                .build())
                        .build())
                .build();
        ToolConfiguration toolConfig = ToolConfiguration.builder()
                .tools(Tool.builder().toolSpec(toolSpec).build())
                .build();
        String toolDefinitionsJson = "[{\"toolSpec\":{\"name\":\"get_weather\",\"description\":\"Get the current weather\"," +
                "\"inputSchema\":{\"json\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"," +
                "\"description\":\"City name\"}},\"required\":[\"location\"]}}}}]";
        Span span = tracer.spanBuilder("chat " + modelId).startSpan();
        try {
            try (var scope = span.makeCurrent()) {
                span.setAttribute(stringKey("gen_ai.operation.name"), "chat");
                span.setAttribute(stringKey("gen_ai.provider.name"), "aws.bedrock");
                span.setAttribute(stringKey("gen_ai.request.model"), modelId);
                span.setAttribute(stringKey("gen_ai.tool.definitions"), toolDefinitionsJson);
                if (mockEndpoint.getHost() != null) {
                    span.setAttribute(stringKey("server.address"), mockEndpoint.getHost());
                }
                if (mockEndpoint.getPort() != -1) {
                    span.setAttribute(longKey("server.port"), (long) mockEndpoint.getPort());
                }

                ConverseResponse response = client.converse(ConverseRequest.builder()
                        .modelId(modelId)
                        .messages(Message.builder()
                                .role(ConversationRole.USER)
                                .content(ContentBlock.fromText("What's the weather in Seattle?"))
                                .build())
                        .toolConfig(toolConfig)
                        .build());

                span.setAttribute(stringArrayKey("gen_ai.response.finish_reasons"),
                        List.of(response.stopReason().toString()));
                span.setAttribute(longKey("gen_ai.usage.input_tokens"), (long) response.usage().inputTokens());
                span.setAttribute(longKey("gen_ai.usage.output_tokens"), (long) response.usage().outputTokens());

                List<ContentBlock> content = response.output().message().content();
                if (!content.isEmpty() && content.get(0).toolUse() != null) {
                    System.out.println("    -> tool_call: " + content.get(0).toolUse().name());
                } else {
                    String text = content.get(0).text();
                    System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
                }
            }
        } finally {
            span.end();
        }
    }
}
