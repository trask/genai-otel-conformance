// Conformance test: OTel contrib opentelemetry-aws-sdk-2.2 instrumentation.
//
// Exercises: converse (Bedrock Converse API)
// against a mock Bedrock server, with the OTel AWS SDK Java library instrumentation.

package com.example.bedrocktest;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.exporter.otlp.logs.OtlpGrpcLogRecordExporter;
import io.opentelemetry.exporter.otlp.metrics.OtlpGrpcMetricExporter;
import io.opentelemetry.instrumentation.awssdk.v2_2.AwsSdkTelemetry;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.logs.SdkLoggerProvider;
import io.opentelemetry.sdk.logs.export.BatchLogRecordProcessor;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.export.PeriodicMetricReader;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
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
import java.time.Duration;

public class AwsBedrockOtelContribTest {

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");
        String otlpEndpoint = System.getenv("OTEL_EXPORTER_OTLP_ENDPOINT");

        System.out.println("=== OTel Contrib: AWS Bedrock Java Conformance Test ===");

        // Set up OTel SDK
        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(BatchSpanProcessor.builder(
                        OtlpGrpcSpanExporter.builder().setEndpoint(otlpEndpoint).build()
                ).build())
                .build();

        SdkLoggerProvider loggerProvider = SdkLoggerProvider.builder()
                .addLogRecordProcessor(BatchLogRecordProcessor.builder(
                        OtlpGrpcLogRecordExporter.builder().setEndpoint(otlpEndpoint).build()
                ).build())
                .build();

        SdkMeterProvider meterProvider = SdkMeterProvider.builder()
                .registerMetricReader(PeriodicMetricReader.builder(
                        OtlpGrpcMetricExporter.builder().setEndpoint(otlpEndpoint).build()
                ).setInterval(Duration.ofSeconds(5)).build())
                .build();

        OpenTelemetrySdk openTelemetry = OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .setLoggerProvider(loggerProvider)
                .setMeterProvider(meterProvider)
                .build();

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

        // Flush and shutdown
        System.out.println("Flushing telemetry...");
        tracerProvider.forceFlush();
        loggerProvider.forceFlush();
        meterProvider.forceFlush();
        tracerProvider.close();
        loggerProvider.close();
        meterProvider.close();
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
