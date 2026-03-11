// Conformance test: Microsoft.Extensions.AI native OTel instrumentation.
//
// Exercises: chat completion via Microsoft.Extensions.AI with UseOpenTelemetry() middleware.
// Points at a mock OpenAI server.

using System;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using OpenAI;
using OpenTelemetry;
using OpenTelemetry.Logs;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

class Program
{
    static async Task Main(string[] args)
    {
        var mockBaseUrl = Environment.GetEnvironmentVariable("MOCK_LLM_URL")! + "/v1";
        var otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")!;

        Console.WriteLine("=== Native: Microsoft.Extensions.AI Conformance Test ===");

        // Configure OTel
        var resourceBuilder = ResourceBuilder.CreateDefault()
            .AddService("extensions-ai-conformance-test");

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

        // Create OpenAI-backed chat client with UseOpenTelemetry() middleware
        IChatClient client = new ChatClientBuilder(
                new OpenAI.Chat.ChatClient(
                    model: "gpt-4o-mini",
                    credential: new System.ClientModel.ApiKeyCredential("mock-key"),
                    options: new OpenAIClientOptions { Endpoint = new Uri(mockBaseUrl) })
                .AsIChatClient())
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
}
