// Conformance test: Spring AI native OTel instrumentation.
//
// Uses Spring AI with built-in Micrometer/OTel tracing pointed at mock server.

package com.example.springaitest;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class SpringAiTestApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringAiTestApplication.class, args);
    }

    @Bean
    CommandLineRunner run(ChatClient.Builder chatClientBuilder) {
        return args -> {
            System.out.println("=== Native: Spring AI Conformance Test ===");

            ChatClient chatClient = chatClientBuilder.build();

            // Scenario: basic chat
            System.out.println("  [chat] basic chat completion");
            String response = chatClient.prompt()
                    .user("Say hello.")
                    .call()
                    .content();
            System.out.println("    -> " + (response != null ? response.substring(0, Math.min(60, response.length())) : "null"));

            // Scenario: streaming chat
            System.out.println("  [chat_streaming] streaming chat completion");
            StringBuilder text = new StringBuilder();
            chatClient.prompt()
                    .user("Tell me a joke.")
                    .stream()
                    .content()
                    .doOnNext(text::append)
                    .blockLast();
            System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));

            System.out.println("Done.");
        };
    }
}
