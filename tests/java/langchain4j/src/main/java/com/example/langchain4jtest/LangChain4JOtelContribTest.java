// Conformance test: OTel contrib instrumentation for LangChain4J.
//
// Exercises: chat, chat_streaming
// against a mock OpenAI server, with OTel SDK registered globally.

package com.example.langchain4jtest;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.output.Response;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.exporter.otlp.logs.OtlpGrpcLogRecordExporter;
import io.opentelemetry.exporter.otlp.metrics.OtlpGrpcMetricExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.logs.SdkLoggerProvider;
import io.opentelemetry.sdk.logs.export.BatchLogRecordProcessor;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.export.PeriodicMetricReader;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class LangChain4JOtelContribTest {

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL") + "/v1";
        String otlpEndpoint = System.getenv("OTEL_EXPORTER_OTLP_ENDPOINT");

        System.out.println("=== OTel Contrib: LangChain4J Conformance Test ===");

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
                .buildAndRegisterGlobal();

        // Create LangChain4J OpenAI chat models pointed at mock server
        OpenAiChatModel chatModel = OpenAiChatModel.builder()
                .baseUrl(mockBaseUrl)
                .apiKey("mock-key")
                .modelName("gpt-4o-mini")
                .build();

        OpenAiStreamingChatModel streamingModel = OpenAiStreamingChatModel.builder()
                .baseUrl(mockBaseUrl)
                .apiKey("mock-key")
                .modelName("gpt-4o-mini")
                .build();

        // Run scenarios
        runChat(chatModel);
        runChatStreaming(streamingModel);

        // Flush and shutdown
        System.out.println("Flushing telemetry...");
        tracerProvider.forceFlush();
        loggerProvider.forceFlush();
        meterProvider.forceFlush();
        tracerProvider.close();
        loggerProvider.close();
        meterProvider.close();
        System.out.println("Done.");
    }

    static void runChat(OpenAiChatModel model) {
        System.out.println("  [chat] basic chat completion");
        String response = model.generate("Say hello.");
        System.out.println("    -> " + response.substring(0, Math.min(60, response.length())));
    }

    static void runChatStreaming(OpenAiStreamingChatModel model) throws Exception {
        System.out.println("  [chat_streaming] streaming chat completion");
        CompletableFuture<String> future = new CompletableFuture<>();
        StringBuilder text = new StringBuilder();
        model.generate("Tell me a joke.", new StreamingResponseHandler<AiMessage>() {
            @Override
            public void onNext(String token) {
                text.append(token);
            }

            @Override
            public void onComplete(Response<AiMessage> response) {
                future.complete(text.toString());
            }

            @Override
            public void onError(Throwable error) {
                future.completeExceptionally(error);
            }
        });
        String result = future.get(30, TimeUnit.SECONDS);
        System.out.println("    -> " + result.substring(0, Math.min(60, result.length())));
    }
}
