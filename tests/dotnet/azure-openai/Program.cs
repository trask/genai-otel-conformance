// Conformance test: Microsoft.Extensions.AI native OTel instrumentation with Azure OpenAI.
//
// Exercises: chat completion, streaming chat completion.
// Points at a mock Azure OpenAI server.
// Supports "native" and "prototype" ecosystems via CONFORMANCE_ECOSYSTEM.

using System;
using System.ClientModel;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Logs;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

class Program
{
    static async Task Main(string[] args)
    {
        var ecosystem = Environment.GetEnvironmentVariable("CONFORMANCE_ECOSYSTEM") ?? "native";

        if (ecosystem == "prototype")
            await RunPrototype();
        else
            await RunNative();
    }

    static async Task RunNative()
    {
        var mockBaseUrl = Environment.GetEnvironmentVariable("MOCK_LLM_URL")!;
        var otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")!;

        Console.WriteLine("=== Native: Azure OpenAI (Extensions.AI) Conformance Test ===");

        // Configure OTel
        var resourceBuilder = ResourceBuilder.CreateDefault()
            .AddService("azure-openai-conformance-test");

        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(resourceBuilder)
            .AddSource("Microsoft.Extensions.AI")
            .AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint))
            .Build();

        using var meterProvider = Sdk.CreateMeterProviderBuilder()
            .SetResourceBuilder(resourceBuilder)
            .AddMeter("Microsoft.Extensions.AI")
            .AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint))
            .Build();

        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddOpenTelemetry(logging =>
            {
                logging.SetResourceBuilder(resourceBuilder);
                logging.AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint));
                logging.IncludeFormattedMessage = true;
            });
        });

        // Create Azure OpenAI client pointing at mock server
        var azureClient = new AzureOpenAIClient(
            new Uri(mockBaseUrl),
            new ApiKeyCredential("mock-key"));

        IChatClient client = new ChatClientBuilder(
                azureClient.GetChatClient("gpt-4o-mini").AsIChatClient())
            .UseOpenTelemetry(loggerFactory: loggerFactory, sourceName: "Microsoft.Extensions.AI")
            .Build();

        // Scenario: basic chat
        Console.WriteLine("  [chat] basic chat completion");
        var response = await client.GetResponseAsync("Say hello.");
        var content = response.Text ?? "";
        Console.WriteLine($"    -> {content[..Math.Min(60, content.Length)]}");

        // Scenario: streaming chat
        Console.WriteLine("  [chat_streaming] streaming chat completion");
        var text = "";
        await foreach (var update in client.GetStreamingResponseAsync("Tell me a joke."))
        {
            text += update.Text;
        }
        Console.WriteLine($"    -> {text[..Math.Min(60, text.Length)]}");

        Console.WriteLine("Flushing telemetry...");
        tracerProvider.ForceFlush();
        meterProvider.ForceFlush();
        loggerFactory.Dispose();
        Console.WriteLine("Done.");
    }

    static async Task RunPrototype()
    {
        var mockBaseUrl = Environment.GetEnvironmentVariable("MOCK_LLM_URL")!;
        var otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")!;

        Console.WriteLine("=== Prototype: Azure OpenAI Conformance Test ===");

        var resourceBuilder = ResourceBuilder.CreateDefault()
            .AddService("azure-openai-conformance-test");

        var activitySource = new ActivitySource("gen_ai.prototype");

        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(resourceBuilder)
            .AddSource("gen_ai.prototype")
            .AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint))
            .Build();

        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddOpenTelemetry(logging =>
            {
                logging.IncludeScopes = true;
                logging.SetResourceBuilder(resourceBuilder);
                logging.AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint));
            });
        });
        var eventLogger = loggerFactory.CreateLogger("gen_ai.prototype");

        // Create Azure OpenAI client pointing at mock server - NO Extensions.AI wrapper
        var modelId = "gpt-4o-mini";
        var azureClient = new AzureOpenAIClient(
            new Uri(mockBaseUrl),
            new ApiKeyCredential("mock-key"));
        var chatClient = azureClient.GetChatClient(modelId);

        // Scenario: basic chat
        Console.WriteLine("  [chat] basic chat completion");
        using (var activity = activitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            var userMessage = "Say hello.";
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            OpenAI.Chat.ChatCompletion completion = await chatClient.CompleteChatAsync(
                new OpenAI.Chat.UserChatMessage(userMessage));

            activity?.SetTag("gen_ai.response.id", completion.Id);
            activity?.SetTag("gen_ai.response.model", completion.Model);
            activity?.SetTag("gen_ai.response.finish_reasons",
                new[] { completion.FinishReason.ToString() });
            if (completion.Usage != null)
            {
                activity?.SetTag("gen_ai.usage.input_tokens", completion.Usage.InputTokenCount);
                activity?.SetTag("gen_ai.usage.output_tokens", completion.Usage.OutputTokenCount);
            }

            var content = completion.Content[0].Text;

            // Emit inference operation details event
            var inputMessagesJson = JsonSerializer.Serialize(new[] {
                new { role = "user", parts = new[] { new { type = "text", content = userMessage } } }
            });
            var outputMessagesJson = JsonSerializer.Serialize(new[] {
                new {
                    role = "assistant",
                    parts = new[] { new { type = "text", content } },
                    finish_reason = completion.FinishReason.ToString()
                }
            });
            using (eventLogger.BeginScope(new Dictionary<string, object>
            {
                ["gen_ai.operation.name"] = "chat",
                ["gen_ai.request.model"] = modelId,
                ["gen_ai.response.id"] = completion.Id,
                ["gen_ai.response.model"] = completion.Model,
                ["gen_ai.response.finish_reasons"] = new[] { completion.FinishReason.ToString() },
                ["gen_ai.usage.input_tokens"] = completion.Usage?.InputTokenCount,
                ["gen_ai.usage.output_tokens"] = completion.Usage?.OutputTokenCount,
                ["gen_ai.input.messages"] = inputMessagesJson,
                ["gen_ai.output.messages"] = outputMessagesJson,
                ["server.address"] = endpoint.Host,
                ["server.port"] = endpoint.Port,
            }))
            {
                eventLogger.LogInformation(new EventId(0, "gen_ai.client.inference.operation.details"), "Inference operation details");
            }

            Console.WriteLine($"    -> {content[..Math.Min(60, content.Length)]}");
        }

        // Scenario: streaming chat
        Console.WriteLine("  [chat_streaming] streaming chat completion");
        using (var activity = activitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            var text = "";

            await foreach (var update in chatClient.CompleteChatStreamingAsync(
                new OpenAI.Chat.UserChatMessage("Tell me a joke.")))
            {
                foreach (var part in update.ContentUpdate)
                    text += part.Text;
                if (update.FinishReason != null)
                    activity?.SetTag("gen_ai.response.finish_reasons",
                        new[] { update.FinishReason.ToString() });
                if (update.Usage != null)
                {
                    activity?.SetTag("gen_ai.usage.input_tokens", update.Usage.InputTokenCount);
                    activity?.SetTag("gen_ai.usage.output_tokens", update.Usage.OutputTokenCount);
                }
            }

            Console.WriteLine($"    -> {text[..Math.Min(60, text.Length)]}");
        }

        // Scenario: chat with tool calling
        Console.WriteLine("  [chat_tool_call] chat with tool calling");
        using (var activity = activitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            var toolName = "get_weather";
            var toolDescription = "Get the current weather";
            var toolParameters = BinaryData.FromString(
                """
                {
                  "type": "object",
                  "properties": {
                    "location": {
                      "type": "string",
                      "description": "City name"
                    }
                  },
                  "required": ["location"]
                }
                """
            );
            var weatherTool = OpenAI.Chat.ChatTool.CreateFunctionTool(
                functionName: toolName,
                functionDescription: toolDescription,
                functionParameters: toolParameters
            );

            List<OpenAI.Chat.ChatMessage> messages =
            [
                new OpenAI.Chat.UserChatMessage("What's the weather in Seattle?"),
            ];
            OpenAI.Chat.ChatCompletionOptions options = new()
            {
                ToolChoice = OpenAI.Chat.ChatToolChoice.CreateAutoChoice(),
            };
            options.Tools.Add(weatherTool);

            var toolDefinitionsJson = JsonSerializer.Serialize(
                options.Tools.Select(t => new Dictionary<string, object>
                {
                    ["type"] = "function",
                    ["name"] = t.FunctionName,
                    ["description"] = t.FunctionDescription,
                    ["parameters"] = JsonSerializer.Deserialize<JsonElement>(t.FunctionParameters)
                }).ToArray()
            );
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("gen_ai.tool.definitions", toolDefinitionsJson);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            OpenAI.Chat.ChatCompletion completion = await chatClient.CompleteChatAsync(messages, options);

            activity?.SetTag("gen_ai.response.id", completion.Id);
            activity?.SetTag("gen_ai.response.model", completion.Model);
            activity?.SetTag("gen_ai.response.finish_reasons",
                new[] { completion.FinishReason.ToString() });
            if (completion.Usage != null)
            {
                activity?.SetTag("gen_ai.usage.input_tokens", completion.Usage.InputTokenCount);
                activity?.SetTag("gen_ai.usage.output_tokens", completion.Usage.OutputTokenCount);
            }

            if (completion.ToolCalls.Count > 0)
            {
                Console.WriteLine($"    -> tool_call: {completion.ToolCalls[0].FunctionName}");
            }
            else
            {
                var content = completion.Content[0].Text;
                Console.WriteLine($"    -> {content[..Math.Min(60, content.Length)]}");
            }
        }

        Console.WriteLine("Flushing telemetry...");
        tracerProvider.ForceFlush();
        loggerFactory.Dispose();
        Console.WriteLine("Done.");
    }
}
