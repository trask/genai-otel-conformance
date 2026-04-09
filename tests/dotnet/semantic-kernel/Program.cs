// Conformance test: Semantic Kernel native OTel instrumentation.
//
// Exercises: chat completion, streaming chat, agent with tool calling.
// Points at a mock OpenAI server.
// Supports "native" and "prototype" ecosystems via CONFORMANCE_ECOSYSTEM.
//
// NOTE: The [agent] scenario exercises agent-style automatic function calling
// via FunctionChoiceBehavior.Auto() with a WeatherPlugin tool. However,
// Semantic Kernel's built-in OTel instrumentation currently only emits
// inference-level chat spans (gen_ai.chat), not gen_ai.agent.* semantic
// convention attributes (e.g., gen_ai.agent.name, gen_ai.agent.description).
// This is a known gap — the agent test exists to track when SK adds
// agent-specific span emission conforming to the OpenTelemetry GenAI
// semantic conventions.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using OpenTelemetry;
using OpenTelemetry.Logs;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

class Program
{
    internal static readonly ActivitySource s_manualActivitySource = new("gen_ai.prototype");

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
        AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics", true);
        AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnosticsSensitive", true);

        var mockBaseUrl = Environment.GetEnvironmentVariable("MOCK_LLM_URL")! + "/v1";
        var otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")!;

        Console.WriteLine("=== Native: Semantic Kernel Conformance Test ===");

        // Configure OTel
        var resourceBuilder = ResourceBuilder.CreateDefault()
            .AddService("semantic-kernel-conformance-test");

        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(resourceBuilder)
            .AddSource("Microsoft.SemanticKernel*")
            .AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint))
            .Build();

        using var meterProvider = Sdk.CreateMeterProviderBuilder()
            .SetResourceBuilder(resourceBuilder)
            .AddMeter("Microsoft.SemanticKernel*")
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

        // Build kernel pointing at mock server
        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(
            modelId: "gpt-4o-mini",
            apiKey: "mock-key",
            httpClient: new System.Net.Http.HttpClient { BaseAddress = new Uri(mockBaseUrl) }
        );
        builder.Services.AddSingleton(loggerFactory);

        var kernel = builder.Build();
        var chatService = kernel.GetRequiredService<IChatCompletionService>();

        // Scenario: basic chat
        Console.WriteLine("  [chat] basic chat completion");
        var history = new ChatHistory();
        history.AddUserMessage("Say hello.");
        var result = await chatService.GetChatMessageContentsAsync(history);
        Console.WriteLine($"    -> {result[0].Content?[..Math.Min(60, result[0].Content?.Length ?? 0)]}");

        // Scenario: streaming chat
        Console.WriteLine("  [chat_streaming] streaming chat completion");
        history.Clear();
        history.AddUserMessage("Tell me a joke.");
        var text = "";
        await foreach (var chunk in chatService.GetStreamingChatMessageContentsAsync(history))
        {
            text += chunk.Content;
        }
        Console.WriteLine($"    -> {text[..Math.Min(60, text.Length)]}");

        // Scenario: agent with auto function calling
        // NOTE: SK emits chat-level spans here, not gen_ai.agent.* spans.
        // This scenario tracks when Semantic Kernel adds agent-specific OTel attributes.
        Console.WriteLine("  [agent] agent with tool calling");
        kernel.ImportPluginFromType<WeatherPlugin>();
        var agentSettings = new OpenAIPromptExecutionSettings
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };
        var agentHistory = new ChatHistory();
        agentHistory.AddUserMessage("What's the weather in Seattle?");
        var agentResult = await chatService.GetChatMessageContentsAsync(agentHistory, agentSettings, kernel);
        Console.WriteLine($"    -> {agentResult[^1].Content?[..Math.Min(60, agentResult[^1].Content?.Length ?? 0)]}");

        Console.WriteLine("Flushing telemetry...");
        tracerProvider.ForceFlush();
        meterProvider.ForceFlush();
        Console.WriteLine("Done.");
    }

    static async Task RunPrototype()
    {
        var mockBaseUrl = Environment.GetEnvironmentVariable("MOCK_LLM_URL")! + "/v1";
        var otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")!;

        Console.WriteLine("=== Prototype: Semantic Kernel Conformance Test ===");

        var resourceBuilder = ResourceBuilder.CreateDefault()
            .AddService("semantic-kernel-conformance-test");

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

        // Build kernel pointing at mock server - NO OTel diagnostic switches
        var builder = Kernel.CreateBuilder();
        var modelId = "gpt-4o-mini";
        builder.AddOpenAIChatCompletion(
            modelId: modelId,
            apiKey: "mock-key",
            httpClient: new System.Net.Http.HttpClient { BaseAddress = new Uri(mockBaseUrl) }
        );

        var kernel = builder.Build();
        var chatService = kernel.GetRequiredService<IChatCompletionService>();

        // Scenario: basic chat
        Console.WriteLine("  [chat] basic chat completion");
        using (var activity = s_manualActivitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            var history = new ChatHistory();
            var userMessage = "Say hello.";
            history.AddUserMessage(userMessage);
            var result = await chatService.GetChatMessageContentsAsync(history);

            var msg = result[0];
            if (msg.ModelId != null) activity?.SetTag("gen_ai.response.model", msg.ModelId);
            var metadata = msg.Metadata;
            string responseId = null;
            string finishReason = null;
            int? inputTokens = null;
            int? outputTokens = null;
            int? reasoningTokens = null;
            if (metadata != null)
            {
                if (metadata.TryGetValue("Id", out var id) && id != null)
                {
                    responseId = id.ToString();
                    activity?.SetTag("gen_ai.response.id", responseId);
                }
                if (metadata.TryGetValue("FinishReason", out var fr) && fr != null)
                {
                    finishReason = fr.ToString();
                    activity?.SetTag("gen_ai.response.finish_reasons",
                        new[] { finishReason });
                }
                if (metadata.TryGetValue("Usage", out var usageObj) &&
                    usageObj is OpenAI.Chat.ChatTokenUsage usage)
                {
                    inputTokens = usage.InputTokenCount;
                    outputTokens = usage.OutputTokenCount;
                    activity?.SetTag("gen_ai.usage.input_tokens", inputTokens);
                    activity?.SetTag("gen_ai.usage.output_tokens", outputTokens);
                    if (usage.OutputTokenDetails?.ReasoningTokenCount > 0)
                    {
                        reasoningTokens = usage.OutputTokenDetails.ReasoningTokenCount;
                        activity?.SetTag("gen_ai.usage.reasoning.output_tokens", reasoningTokens);
                    }
                }
            }

            // Emit inference operation details event
            var inputMessagesJson = JsonSerializer.Serialize(new[] {
                new { role = "user", parts = new[] { new { type = "text", content = userMessage } } }
            });
            var outputMessagesJson = JsonSerializer.Serialize(new[] {
                new {
                    role = "assistant",
                    parts = new[] { new { type = "text", content = msg.Content } },
                    finish_reason = finishReason
                }
            });
            using (eventLogger.BeginScope(new Dictionary<string, object>
            {
                ["gen_ai.operation.name"] = "chat",
                ["gen_ai.request.model"] = modelId,
                ["gen_ai.response.id"] = responseId,
                ["gen_ai.response.model"] = msg.ModelId,
                ["gen_ai.response.finish_reasons"] = finishReason != null ? new[] { finishReason } : null,
                ["gen_ai.usage.input_tokens"] = inputTokens,
                ["gen_ai.usage.output_tokens"] = outputTokens,
                ["gen_ai.usage.reasoning.output_tokens"] = reasoningTokens,
                ["gen_ai.input.messages"] = inputMessagesJson,
                ["gen_ai.output.messages"] = outputMessagesJson,
                ["server.address"] = endpoint.Host,
                ["server.port"] = endpoint.Port,
            }))
            {
                eventLogger.LogInformation(new EventId(0, "gen_ai.client.inference.operation.details"), "Inference operation details");
            }

            Console.WriteLine($"    -> {msg.Content?[..Math.Min(60, msg.Content?.Length ?? 0)]}");
        }

        // Scenario: streaming chat
        Console.WriteLine("  [chat_streaming] streaming chat completion");
        using (var activity = s_manualActivitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            var history = new ChatHistory();
            history.AddUserMessage("Tell me a joke.");
            var text = "";
            await foreach (var chunk in chatService.GetStreamingChatMessageContentsAsync(history))
            {
                text += chunk.Content;
                if (chunk.ModelId != null) activity?.SetTag("gen_ai.response.model", chunk.ModelId);
                var metadata = chunk.Metadata;
                if (metadata != null)
                {
                    if (metadata.TryGetValue("Id", out var id) && id != null)
                        activity?.SetTag("gen_ai.response.id", id.ToString());
                    if (metadata.TryGetValue("FinishReason", out var fr) && fr != null)
                        activity?.SetTag("gen_ai.response.finish_reasons",
                            new[] { fr.ToString() });
                    if (metadata.TryGetValue("Usage", out var usageObj) &&
                        usageObj is OpenAI.Chat.ChatTokenUsage usage)
                    {
                        activity?.SetTag("gen_ai.usage.input_tokens", usage.InputTokenCount);
                        activity?.SetTag("gen_ai.usage.output_tokens", usage.OutputTokenCount);
                        if (usage.OutputTokenDetails?.ReasoningTokenCount > 0)
                            activity?.SetTag("gen_ai.usage.reasoning.output_tokens", usage.OutputTokenDetails.ReasoningTokenCount);
                    }
                }
            }

            Console.WriteLine($"    -> {text[..Math.Min(60, text.Length)]}");
        }

        // Scenario: agent with auto function calling
        Console.WriteLine("  [agent] agent with tool calling");
        kernel.ImportPluginFromType<PrototypeWeatherPlugin>();
        using (var activity = s_manualActivitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            var toolDefinitionsJson = JsonSerializer.Serialize(
                kernel.Plugins
                    .SelectMany(p => p)
                    .Select(f => new Dictionary<string, object>
                    {
                        ["type"] = "function",
                        ["name"] = f.Name,
                        ["description"] = f.Description,
                        ["parameters"] = new
                        {
                            type = "object",
                            properties = f.Metadata.Parameters.ToDictionary(
                                p => p.Name,
                                p => new { type = ToJsonSchemaType(p.ParameterType) }),
                            required = f.Metadata.Parameters
                                .Where(p => p.IsRequired)
                                .Select(p => p.Name)
                        }
                    }));
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("gen_ai.tool.definitions", toolDefinitionsJson);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            var agentSettings = new OpenAIPromptExecutionSettings
            {
                FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
            };
            var agentHistory = new ChatHistory();
            agentHistory.AddUserMessage("What's the weather in Seattle?");
            var agentResult = await chatService.GetChatMessageContentsAsync(
                agentHistory, agentSettings, kernel);

            var lastMsg = agentResult[^1];
            if (lastMsg.ModelId != null) activity?.SetTag("gen_ai.response.model", lastMsg.ModelId);
            var metadata = lastMsg.Metadata;
            if (metadata != null)
            {
                if (metadata.TryGetValue("Id", out var id) && id != null)
                    activity?.SetTag("gen_ai.response.id", id.ToString());
                if (metadata.TryGetValue("FinishReason", out var fr) && fr != null)
                    activity?.SetTag("gen_ai.response.finish_reasons",
                        new[] { fr.ToString() });
                if (metadata.TryGetValue("Usage", out var usageObj) &&
                    usageObj is OpenAI.Chat.ChatTokenUsage usage)
                {
                    activity?.SetTag("gen_ai.usage.input_tokens", usage.InputTokenCount);
                    activity?.SetTag("gen_ai.usage.output_tokens", usage.OutputTokenCount);
                    if (usage.OutputTokenDetails?.ReasoningTokenCount > 0)
                        activity?.SetTag("gen_ai.usage.reasoning.output_tokens", usage.OutputTokenDetails.ReasoningTokenCount);
                }
            }

            Console.WriteLine($"    -> {lastMsg.Content?[..Math.Min(60, lastMsg.Content?.Length ?? 0)]}");
        }

        Console.WriteLine("Flushing telemetry...");
        tracerProvider.ForceFlush();
        loggerFactory.Dispose();
        Console.WriteLine("Done.");
    }

    static string ToJsonSchemaType(Type type)
    {
        if (type == null) return "string";
        type = Nullable.GetUnderlyingType(type) ?? type;
        if (type == typeof(string)) return "string";
        if (type == typeof(bool)) return "boolean";
        if (type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(byte)) return "integer";
        if (type == typeof(float) || type == typeof(double) || type == typeof(decimal)) return "number";
        if (type.IsArray || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>))) return "array";
        return "string";
    }
}

class WeatherPlugin
{
    [KernelFunction, Description("Get the current weather for a location")]
    public string GetWeather(string location) => "Sunny, 72°F";
}

class PrototypeWeatherPlugin
{
    [KernelFunction, Description("Get the current weather for a location")]
    public string GetWeather(string location)
    {
        using var activity = Program.s_manualActivitySource.StartActivity("execute_tool get_weather");
        activity?.SetTag("gen_ai.operation.name", "execute_tool");
        activity?.SetTag("gen_ai.tool.name", "get_weather");
        activity?.SetTag("gen_ai.tool.description", "Get the current weather for a location");
        activity?.SetTag("gen_ai.tool.call.arguments", $"{{\"location\":\"{location}\"}}");
        var result = "Sunny, 72°F";
        activity?.SetTag("gen_ai.tool.call.result", result);
        return result;
    }
}
