// Conformance test: OTel contrib opentelemetry-aws-sdk-2.2 instrumentation.
//
// Exercises: invoke_agent (Bedrock Agent Runtime InvokeAgent API)
// against a mock Bedrock Agent server, with the OTel AWS SDK Java library instrumentation.

package com.example.bedrockagenttest;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.instrumentation.awssdk.v2_2.AwsSdkTelemetry;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.client.config.ClientOverrideConfiguration;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockagentruntime.BedrockAgentRuntimeClient;
import software.amazon.awssdk.services.bedrockagentruntime.model.InvokeAgentRequest;
import software.amazon.awssdk.services.bedrockagentruntime.model.InvokeAgentResponseHandler;

import java.net.URI;
import java.util.concurrent.atomic.AtomicReference;

public class AwsBedrockAgentOtelContribTest {

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");

        System.out.println("=== OTel Contrib: AWS Bedrock Agent Java Conformance Test ===");

        OpenTelemetry openTelemetry = GlobalOpenTelemetry.get();

        // Create AWS SDK OTel instrumentation
        AwsSdkTelemetry telemetry = AwsSdkTelemetry.builder(openTelemetry).build();

        // Create instrumented Bedrock Agent Runtime client pointing at mock server
        BedrockAgentRuntimeClient client = BedrockAgentRuntimeClient.builder()
                .endpointOverride(new URI(mockBaseUrl))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .overrideConfiguration(ClientOverrideConfiguration.builder()
                        .addExecutionInterceptor(telemetry.createExecutionInterceptor())
                        .build())
                .build();

        // Run scenarios
        runInvokeAgent(client);

        client.close();
        System.out.println("Done.");
    }

    static void runInvokeAgent(BedrockAgentRuntimeClient client) {
        System.out.println("  [invoke_agent] Bedrock Agent Runtime InvokeAgent");

        AtomicReference<String> completionText = new AtomicReference<>("");

        client.invokeAgent(
                InvokeAgentRequest.builder()
                        .agentId("MOCK_AGENT_ID")
                        .agentAliasId("MOCK_ALIAS_ID")
                        .sessionId("mock-session-001")
                        .inputText("Say hello.")
                        .build(),
                InvokeAgentResponseHandler.builder()
                        .onResponse(response -> {})
                        .subscriber(event -> {
                            event.accept(InvokeAgentResponseHandler.Visitor.builder()
                                    .onChunk(chunk -> {
                                        String text = chunk.bytes().asUtf8String();
                                        completionText.set(completionText.get() + text);
                                    })
                                    .build());
                        })
                        .build()
        );

        String text = completionText.get();
        System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
    }
}
