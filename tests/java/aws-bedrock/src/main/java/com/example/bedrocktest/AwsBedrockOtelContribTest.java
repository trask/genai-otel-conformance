// Conformance test: OTel contrib opentelemetry-aws-sdk-2.2 instrumentation.
//
// Exercises: converse (Bedrock Converse API)
// against a mock Bedrock server, with the OTel AWS SDK Java library instrumentation.

package com.example.bedrocktest;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.instrumentation.awssdk.v2_2.AwsSdkTelemetry;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.client.config.ClientOverrideConfiguration;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ConversationRole;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseRequest;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseResponse;
import software.amazon.awssdk.services.bedrockruntime.model.Message;

import java.net.URI;

public class AwsBedrockOtelContribTest {

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");

        System.out.println("=== OTel Contrib: AWS Bedrock Java Conformance Test ===");

        OpenTelemetry openTelemetry = GlobalOpenTelemetry.get();

        // Create AWS SDK OTel instrumentation
        AwsSdkTelemetry telemetry = AwsSdkTelemetry.builder(openTelemetry).build();

        // Create instrumented Bedrock client pointing at mock server
        BedrockRuntimeClient client = BedrockRuntimeClient.builder()
                .endpointOverride(new URI(mockBaseUrl))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .overrideConfiguration(ClientOverrideConfiguration.builder()
                        .addExecutionInterceptor(telemetry.createExecutionInterceptor())
                        .build())
                .build();

        // Run scenarios
        runConverse(client);

        client.close();
        System.out.println("Done.");
    }

    static void runConverse(BedrockRuntimeClient client) {
        System.out.println("  [converse] Bedrock Converse API");
        ConverseResponse response = client.converse(ConverseRequest.builder()
                .modelId("anthropic.claude-3-haiku-20240307-v1:0")
                .messages(Message.builder()
                        .role(ConversationRole.USER)
                        .content(ContentBlock.fromText("Say hello."))
                        .build())
                .build());
        String text = response.output().message().content().get(0).text();
        System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
    }
}
