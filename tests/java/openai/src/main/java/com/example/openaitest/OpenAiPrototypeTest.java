// Conformance test: Manual OTel instrumentation for OpenAI Java.
//
// Exercises: chat, chat_streaming, chat_tool_call, embeddings
// against a mock OpenAI server, with manual OTel span creation around raw SDK calls.

package com.example.openaitest;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.openai.core.JsonValue;
import com.openai.core.ObjectMappers;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.http.StreamResponse;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionChunk;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionFunctionTool;
import com.openai.models.chat.completions.ChatCompletionMessageToolCall;
import com.openai.models.embeddings.CreateEmbeddingResponse;
import com.openai.models.embeddings.EmbeddingCreateParams;
import com.openai.models.embeddings.EmbeddingModel;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;

import java.util.List;
import java.util.Map;

import static java.util.Collections.singletonList;

public class OpenAiPrototypeTest {

    private static final Tracer tracer = GlobalOpenTelemetry.getTracer("gen_ai.prototype");

    public static void main(String[] args) {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL") + "/v1";

        System.out.println("=== Prototype: OpenAI Java Conformance Test ===");

        // Create raw client - NO instrumentation wrapper
        OpenAIClient client = OpenAIOkHttpClient.builder()
                .baseUrl(mockBaseUrl)
                .apiKey("mock-key")
                .build();

        // Run scenarios
        runChat(client);
        runChatStreaming(client);
        runChatToolCall(client);
        runEmbeddings(client);

        System.out.println("Done.");
    }

    static void runChat(OpenAIClient client) {
        System.out.println("  [chat] basic chat completion");
        ChatModel requestModel = ChatModel.GPT_4O_MINI;
        Span span = tracer.spanBuilder("chat gpt-4o-mini").startSpan();
        try {
            span.setAttribute("gen_ai.operation.name", "chat");
            span.setAttribute("gen_ai.provider.name", "openai");
            span.setAttribute("gen_ai.request.model", requestModel.toString());

            ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                    .model(requestModel)
                    .addUserMessage("Say hello.")
                    .build();
            ChatCompletion completion = client.chat().completions().create(params);

            span.setAttribute("gen_ai.response.id", completion.id());
            span.setAttribute("gen_ai.response.model", completion.model());
            ChatCompletion.Choice choice = completion.choices().get(0);
            span.setAttribute(AttributeKey.stringArrayKey("gen_ai.response.finish_reasons"),
                    List.of(choice.finishReason().toString()));
            completion.usage().ifPresent(usage -> {
                span.setAttribute("gen_ai.usage.input_tokens", usage.promptTokens());
                span.setAttribute("gen_ai.usage.output_tokens", usage.completionTokens());
            });

            String content = choice.message().content().orElse("");
            System.out.println("    -> " + content.substring(0, Math.min(60, content.length())));
        } finally {
            span.end();
        }
    }

    static void runChatStreaming(OpenAIClient client) {
        System.out.println("  [chat_streaming] streaming chat completion");
        ChatModel requestModel = ChatModel.GPT_4O_MINI;
        Span span = tracer.spanBuilder("chat gpt-4o-mini").startSpan();
        try {
            span.setAttribute("gen_ai.operation.name", "chat");
            span.setAttribute("gen_ai.provider.name", "openai");
            span.setAttribute("gen_ai.request.model", requestModel.toString());

            ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                    .model(requestModel)
                    .addUserMessage("Tell me a joke.")
                    .build();

            StringBuilder text = new StringBuilder();
            String[] responseId = {null};
            String[] responseModel = {null};
            String[] finishReason = {null};

            try (StreamResponse<ChatCompletionChunk> stream =
                         client.chat().completions().createStreaming(params)) {
                stream.stream().forEach(chunk -> {
                    if (responseId[0] == null) responseId[0] = chunk.id();
                    if (responseModel[0] == null) responseModel[0] = chunk.model();
                    for (ChatCompletionChunk.Choice choice : chunk.choices()) {
                        choice.delta().content().ifPresent(text::append);
                        choice.finishReason().ifPresent(fr -> finishReason[0] = fr.toString());
                    }
                    chunk.usage().ifPresent(usage -> {
                        span.setAttribute("gen_ai.usage.input_tokens", usage.promptTokens());
                        span.setAttribute("gen_ai.usage.output_tokens", usage.completionTokens());
                    });
                });
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            if (responseId[0] != null) span.setAttribute("gen_ai.response.id", responseId[0]);
            if (responseModel[0] != null) span.setAttribute("gen_ai.response.model", responseModel[0]);
            if (finishReason[0] != null) {
                span.setAttribute(AttributeKey.stringArrayKey("gen_ai.response.finish_reasons"),
                        List.of(finishReason[0]));
            }

            System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
        } finally {
            span.end();
        }
    }

    static void runChatToolCall(OpenAIClient client) {
        System.out.println("  [chat_tool_call] chat with tool calling");
        ChatModel requestModel = ChatModel.GPT_4O_MINI;
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(requestModel)
                .addUserMessage("What's the weather in Seattle?")
                .addTool(ChatCompletionFunctionTool.builder()
                        .function(FunctionDefinition.builder()
                                .name("get_weather")
                                .description("Get the current weather")
                                .parameters(FunctionParameters.builder()
                                        .putAdditionalProperty("type", JsonValue.from("object"))
                                        .putAdditionalProperty("properties", JsonValue.from(
                                                Map.of("location", Map.of(
                                                        "type", "string",
                                                        "description", "City name"
                                                ))
                                        ))
                                        .putAdditionalProperty("required", JsonValue.from(singletonList("location")))
                                        .build())
                                .build())
                        .build())
                .build();
        // Derive from params.tools() — the same tools passed to the API call
        String toolDefinitionsJson;
        try {
            toolDefinitionsJson = ObjectMappers.jsonMapper().writeValueAsString(params.tools().orElse(List.of()));
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        Span span = tracer.spanBuilder("chat gpt-4o-mini").startSpan();
        try {
            span.setAttribute("gen_ai.operation.name", "chat");
            span.setAttribute("gen_ai.provider.name", "openai");
            span.setAttribute("gen_ai.request.model", requestModel.toString());
            span.setAttribute("gen_ai.tool.definitions", toolDefinitionsJson);
            ChatCompletion completion = client.chat().completions().create(params);

            span.setAttribute("gen_ai.response.id", completion.id());
            span.setAttribute("gen_ai.response.model", completion.model());
            ChatCompletion.Choice choice = completion.choices().get(0);
            span.setAttribute(AttributeKey.stringArrayKey("gen_ai.response.finish_reasons"),
                    List.of(choice.finishReason().toString()));
            completion.usage().ifPresent(usage -> {
                span.setAttribute("gen_ai.usage.input_tokens", usage.promptTokens());
                span.setAttribute("gen_ai.usage.output_tokens", usage.completionTokens());
            });

            List<ChatCompletionMessageToolCall> toolCalls =
                    choice.message().toolCalls().orElse(List.of());
            if (!toolCalls.isEmpty()) {
                System.out.println("    -> tool_call: " + toolCalls.get(0).asFunction().function().name());
            } else {
                String content = choice.message().content().orElse("");
                System.out.println("    -> " + content.substring(0, Math.min(60, content.length())));
            }
        } finally {
            span.end();
        }
    }

    static void runEmbeddings(OpenAIClient client) {
        System.out.println("  [embeddings] embedding generation");
        EmbeddingModel requestModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL;
        EmbeddingCreateParams.EncodingFormat requestEncodingFormat = EmbeddingCreateParams.EncodingFormat.BASE64;
        Span span = tracer.spanBuilder("embeddings text-embedding-3-small").startSpan();
        try {
            span.setAttribute("gen_ai.operation.name", "embeddings");
            span.setAttribute("gen_ai.provider.name", "openai");
            span.setAttribute("gen_ai.request.model", requestModel.toString());
            span.setAttribute(AttributeKey.stringArrayKey("gen_ai.request.encoding_formats"),
                List.of(requestEncodingFormat.asString()));

            EmbeddingCreateParams params = EmbeddingCreateParams.builder()
                    .model(requestModel)
                    .encodingFormat(requestEncodingFormat)
                    .input("Hello, world!")
                    .build();
            CreateEmbeddingResponse response = client.embeddings().create(params);

            span.setAttribute("gen_ai.response.model", response.model());
            span.setAttribute("gen_ai.usage.input_tokens", response.usage().promptTokens());

            System.out.println("    -> embedding dim: " + response.data().get(0).embedding().size());
        } finally {
            span.end();
        }
    }
}
