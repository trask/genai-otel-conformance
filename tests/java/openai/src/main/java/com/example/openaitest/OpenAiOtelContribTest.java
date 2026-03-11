// Conformance test: OTel contrib opentelemetry-openai-java-1.1 instrumentation.
//
// Exercises: chat, chat_streaming, chat_tool_call, embeddings
// against a mock OpenAI server, with the OTel OpenAI Java library instrumentation.

package com.example.openaitest;

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
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.instrumentation.openai.v1_1.OpenAITelemetry;

import java.util.List;
import java.util.Map;

import static java.util.Collections.singletonList;

public class OpenAiOtelContribTest {

    public static void main(String[] args) {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL") + "/v1";

        System.out.println("=== OTel Contrib: OpenAI Java Conformance Test ===");

        OpenTelemetry openTelemetry = GlobalOpenTelemetry.get();

        // Create instrumented client
        OpenAIClient rawClient = OpenAIOkHttpClient.builder()
                .baseUrl(mockBaseUrl)
                .apiKey("mock-key")
                .build();

        OpenAIClient client = OpenAITelemetry.builder(openTelemetry).build().wrap(rawClient);

        // Run scenarios
        runChat(client);
        runChatStreaming(client);
        runChatToolCall(client);
        runEmbeddings(client);

        System.out.println("Done.");
    }

    static void runChat(OpenAIClient client) {
        System.out.println("  [chat] basic chat completion");
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_MINI)
                .addUserMessage("Say hello.")
                .build();
        ChatCompletion completion = client.chat().completions().create(params);
        String content = completion.choices().get(0).message().content().orElse("");
        System.out.println("    -> " + content.substring(0, Math.min(60, content.length())));
    }

    static void runChatStreaming(OpenAIClient client) {
        System.out.println("  [chat_streaming] streaming chat completion");
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_MINI)
                .addUserMessage("Tell me a joke.")
                .build();
        StringBuilder text = new StringBuilder();
        try (StreamResponse<ChatCompletionChunk> stream =
                     client.chat().completions().createStreaming(params)) {
            stream.stream()
                    .flatMap(chunk -> chunk.choices().stream())
                    .flatMap(choice -> choice.delta().content().stream())
                    .forEach(text::append);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
    }

    static void runChatToolCall(OpenAIClient client) {
        System.out.println("  [chat_tool_call] chat with tool calling");
        ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                .model(ChatModel.GPT_4O_MINI)
                .addUserMessage("What's the weather in Seattle?")
                .addTool(ChatCompletionFunctionTool.builder()
                        .function(FunctionDefinition.builder()
                                .name("get_weather")
                                .description("Get the current weather")
                                .parameters(FunctionParameters.builder()
                                        .putAdditionalProperty("type", com.openai.core.JsonValue.from("object"))
                                        .putAdditionalProperty("properties", com.openai.core.JsonValue.from(
                                                Map.of("location", Map.of(
                                                        "type", "string",
                                                        "description", "City name"
                                                ))
                                        ))
                                        .putAdditionalProperty("required", com.openai.core.JsonValue.from(
                                                singletonList("location")
                                        ))
                                        .build())
                                .build())
                        .build())
                .build();
        ChatCompletion completion = client.chat().completions().create(params);
        ChatCompletion.Choice choice = completion.choices().get(0);
        List<ChatCompletionMessageToolCall> toolCalls =
                choice.message().toolCalls().orElse(List.of());
        if (!toolCalls.isEmpty()) {
            System.out.println("    -> tool_call: " + toolCalls.get(0).asFunction().function().name());
        } else {
            String content = choice.message().content().orElse("");
            System.out.println("    -> " + content.substring(0, Math.min(60, content.length())));
        }
    }

    static void runEmbeddings(OpenAIClient client) {
        System.out.println("  [embeddings] embedding generation");
        EmbeddingCreateParams params = EmbeddingCreateParams.builder()
                .model(EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
                .input("Hello, world!")
                .build();
        CreateEmbeddingResponse response = client.embeddings().create(params);
        System.out.println("    -> embedding dim: " + response.data().get(0).embedding().size());
    }
}
