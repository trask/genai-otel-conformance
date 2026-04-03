// Conformance test: Microsoft.Extensions.AI native OTel instrumentation.
//
// Exercises: chat completion via Microsoft.Extensions.AI with UseOpenTelemetry() middleware.
// Points at a mock OpenAI server.
// Supports "native" and "prototype" ecosystems via CONFORMANCE_ECOSYSTEM.

using System;
using System.Diagnostics;
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
        var ecosystem = Environment.GetEnvironmentVariable("CONFORMANCE_ECOSYSTEM") ?? "native";

        if (ecosystem == "prototype")
            await RunPrototype();
        else
            await RunNative();
    }

    static async Task RunNative()
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

    static async Task RunPrototype()
    {
        var mockBaseUrl = Environment.GetEnvironmentVariable("MOCK_LLM_URL")! + "/v1";
        var otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")!;

        Console.WriteLine("=== Prototype: Microsoft.Extensions.AI Conformance Test ===");

        var resourceBuilder = ResourceBuilder.CreateDefault()
            .AddService("extensions-ai-conformance-test");

        var activitySource = new ActivitySource("gen_ai.prototype");

        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(resourceBuilder)
            .AddSource("gen_ai.prototype")
            .AddOtlpExporter(o => o.Endpoint = new Uri(otlpEndpoint))
            .Build();

        // Create raw OpenAI ChatClient - NO Extensions.AI wrapper
        var modelId = "gpt-4o-mini";
        var chatClient = new OpenAI.Chat.ChatClient(
            model: modelId,
            credential: new System.ClientModel.ApiKeyCredential("mock-key"),
            options: new OpenAIClientOptions { Endpoint = new Uri(mockBaseUrl) });

        // Scenario: basic chat
        Console.WriteLine("  [chat] basic chat completion");
        using (var activity = activitySource.StartActivity("chat gpt-4o-mini"))
        {
            var endpoint = new Uri(mockBaseUrl);
            activity?.SetTag("gen_ai.operation.name", "chat");
            activity?.SetTag("gen_ai.provider.name", "openai");
            activity?.SetTag("gen_ai.request.model", modelId);
            activity?.SetTag("server.address", endpoint.Host);
            activity?.SetTag("server.port", endpoint.Port);

            OpenAI.Chat.ChatCompletion completion = await chatClient.CompleteChatAsync(
                new OpenAI.Chat.UserChatMessage("Say hello."));

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

        Console.WriteLine("Flushing telemetry...");
        tracerProvider.ForceFlush();
        Console.WriteLine("Done.");
    }
}
