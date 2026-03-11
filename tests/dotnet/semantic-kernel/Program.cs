// Conformance test: Semantic Kernel native OTel instrumentation.
//
// Exercises: chat completion via Semantic Kernel with built-in OTel tracing.
// Points at a mock OpenAI server.

using System;
using System.ComponentModel;
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
    static async Task Main(string[] args)
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
}

class WeatherPlugin
{
    [KernelFunction, Description("Get the current weather for a location")]
    public string GetWeather(string location) => "Sunny, 72°F";
}
