// Conformance test: Spring AI with OpenInference semantic conventions.
//
// Uses the Spring Boot auto-configured OpenTelemetry (via Micrometer bridge)
// and enriches spans with OpenInference attributes such as openinference.span.kind,
// llm.model_name, llm.input_messages, llm.output_messages, and token counts.

package com.example.springaitest;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class SpringAiOpenInferenceTest {

    // OpenInference semantic convention attribute keys
    static final AttributeKey<String> OPENINFERENCE_SPAN_KIND =
            AttributeKey.stringKey("openinference.span.kind");
    static final AttributeKey<String> INPUT_VALUE =
            AttributeKey.stringKey("input.value");
    static final AttributeKey<String> OUTPUT_VALUE =
            AttributeKey.stringKey("output.value");
    static final AttributeKey<String> LLM_MODEL_NAME =
            AttributeKey.stringKey("llm.model_name");
    static final AttributeKey<String> LLM_INPUT_MESSAGES =
            AttributeKey.stringKey("llm.input_messages");
    static final AttributeKey<String> LLM_OUTPUT_MESSAGES =
            AttributeKey.stringKey("llm.output_messages");
    static final AttributeKey<Long> LLM_TOKEN_COUNT_PROMPT =
            AttributeKey.longKey("llm.token_count.prompt");
    static final AttributeKey<Long> LLM_TOKEN_COUNT_COMPLETION =
            AttributeKey.longKey("llm.token_count.completion");
    static final AttributeKey<Long> LLM_TOKEN_COUNT_TOTAL =
            AttributeKey.longKey("llm.token_count.total");

    public static void main(String[] args) {
        SpringApplication.run(SpringAiOpenInferenceTest.class, args);
    }

    @Bean
    CommandLineRunner run(ChatClient.Builder chatClientBuilder, OpenTelemetry openTelemetry) {
        return args -> {
            System.out.println("=== OpenInference: Spring AI Conformance Test ===");

            Tracer tracer = openTelemetry.getTracer("openinference.spring-ai");
            ChatClient chatClient = chatClientBuilder.build();

            // Scenario 1: basic chat with OpenInference attributes
            runBasicChat(tracer, chatClient);

            // Scenario 2: streaming chat with OpenInference attributes
            runStreamingChat(tracer, chatClient);

            System.out.println("Done.");
        };
    }

    private void runBasicChat(Tracer tracer, ChatClient chatClient) {
        System.out.println("  [chat] basic chat completion with OpenInference spans");
        String userMessage = "Say hello.";

        Span span = tracer.spanBuilder("chat gpt-4o-mini").startSpan();
        try (Scope scope = span.makeCurrent()) {
            span.setAttribute(OPENINFERENCE_SPAN_KIND, "LLM");
            span.setAttribute(LLM_MODEL_NAME, "gpt-4o-mini");
            span.setAttribute(INPUT_VALUE, userMessage);
            span.setAttribute(LLM_INPUT_MESSAGES,
                    jsonMessages("user", userMessage));

            ChatResponse chatResponse = chatClient.prompt()
                    .user(userMessage)
                    .call()
                    .chatResponse();

            String content = chatResponse.getResult().getOutput().getText();
            span.setAttribute(OUTPUT_VALUE, content != null ? content : "");
            span.setAttribute(LLM_OUTPUT_MESSAGES,
                    jsonMessages("assistant", content != null ? content : ""));

            if (chatResponse.getMetadata() != null
                    && chatResponse.getMetadata().getUsage() != null) {
                var usage = chatResponse.getMetadata().getUsage();
                span.setAttribute(LLM_TOKEN_COUNT_PROMPT, (long) usage.getPromptTokens());
                span.setAttribute(LLM_TOKEN_COUNT_COMPLETION, (long) usage.getCompletionTokens());
                span.setAttribute(LLM_TOKEN_COUNT_TOTAL, (long) usage.getTotalTokens());
            }

            System.out.println("    -> " + truncate(content));
        } catch (Exception e) {
            span.setStatus(StatusCode.ERROR, e.getMessage());
            span.recordException(e);
            System.err.println("    -> ERROR: " + e.getMessage());
        } finally {
            span.end();
        }
    }

    private void runStreamingChat(Tracer tracer, ChatClient chatClient) {
        System.out.println("  [chat_streaming] streaming chat with OpenInference spans");
        String userMessage = "Tell me a joke.";

        Span span = tracer.spanBuilder("chat gpt-4o-mini").startSpan();
        try (Scope scope = span.makeCurrent()) {
            span.setAttribute(OPENINFERENCE_SPAN_KIND, "LLM");
            span.setAttribute(LLM_MODEL_NAME, "gpt-4o-mini");
            span.setAttribute(INPUT_VALUE, userMessage);
            span.setAttribute(LLM_INPUT_MESSAGES,
                    jsonMessages("user", userMessage));

            StringBuilder text = new StringBuilder();
            chatClient.prompt()
                    .user(userMessage)
                    .stream()
                    .content()
                    .doOnNext(text::append)
                    .blockLast();

            String content = text.toString();
            span.setAttribute(OUTPUT_VALUE, content);
            span.setAttribute(LLM_OUTPUT_MESSAGES,
                    jsonMessages("assistant", content));

            System.out.println("    -> " + truncate(content));
        } catch (Exception e) {
            span.setStatus(StatusCode.ERROR, e.getMessage());
            span.recordException(e);
            System.err.println("    -> ERROR: " + e.getMessage());
        } finally {
            span.end();
        }
    }

    /** Build a minimal OpenInference-style JSON array for a single message. */
    private static String jsonMessages(String role, String content) {
        String escaped = content.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n");
        return "[{\"message.role\":\"" + role
                + "\",\"message.content\":\"" + escaped + "\"}]";
    }

    private static String truncate(String s) {
        if (s == null) return "null";
        return s.substring(0, Math.min(60, s.length()));
    }
}
