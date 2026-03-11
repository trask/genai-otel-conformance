// Conformance test: Microsoft.Extensions.AI native OTel instrumentation with Azure OpenAI.
//
// Exercises: chat completion, streaming chat completion.
// Points at a mock Azure OpenAI server.

using System;
using System.ClientModel;
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
}
