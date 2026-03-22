// Conformance test: Manual invoke_agent instrumentation for AWS Bedrock Agent.
//
// Exercises: invoke_agent (Bedrock Agent Runtime InvokeAgent API)
// against a mock Bedrock server, with manual OTel spans.

package com.example.bedrockagenttest;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockagentruntime.BedrockAgentRuntimeClient;
import software.amazon.awssdk.services.bedrockagentruntime.model.InvokeAgentRequest;
import software.amazon.awssdk.services.bedrockagentruntime.model.InvokeAgentResponse;
import software.amazon.awssdk.services.bedrockagentruntime.model.InvokeAgentResponseHandler;
import software.amazon.awssdk.services.bedrockagentruntime.BedrockAgentRuntimeAsyncClient;

import java.net.URI;

public class AwsBedrockAgentManualTest {

    private static final String AGENT_ID = "MOCK_AGENT_ID";
    private static final String AGENT_ALIAS_ID = "MOCK_ALIAS_ID";
    private static final String SESSION_ID = "mock-session-001";
    private static final String AGENT_NAME = "conformance-test-agent";

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");

        System.out.println("=== Manual: AWS Bedrock Agent Java Invoke Agent Conformance Test ===");

        URI parsedUri = new URI(mockBaseUrl);
        String serverAddress = parsedUri.getHost();
        int serverPort = parsedUri.getPort() > 0 ? parsedUri.getPort() : 443;

        Tracer tracer = GlobalOpenTelemetry.getTracer("gen_ai.client.aws_bedrock");

        BedrockAgentRuntimeClient client = BedrockAgentRuntimeClient.builder()
                .endpointOverride(new URI(mockBaseUrl))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .build();

        System.out.println("  [invoke_agent] Bedrock Agent Runtime InvokeAgent");
        Span span = tracer.spanBuilder("invoke_agent")
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        try (Scope ignored = span.makeCurrent()) {
            span.setAttribute("gen_ai.operation.name", "invoke_agent");
            span.setAttribute("gen_ai.provider.name", "aws.bedrock");
            span.setAttribute("gen_ai.agent.id", AGENT_ID);
            span.setAttribute("gen_ai.agent.name", AGENT_NAME);
            span.setAttribute("server.address", serverAddress);
            span.setAttribute("server.port", (long) serverPort);

            InvokeAgentResponse response = client.invokeAgent(InvokeAgentRequest.builder()
                    .agentId(AGENT_ID)
                    .agentAliasId(AGENT_ALIAS_ID)
                    .sessionId(SESSION_ID)
                    .inputText("Hello, agent!")
                    .build());

            StringBuilder text = new StringBuilder();
            response.completion().forEach(event -> {
                if (event.chunk() != null && event.chunk().bytes() != null) {
                    text.append(event.chunk().bytes().asUtf8String());
                }
            });
            System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
        } catch (Exception exc) {
            span.setStatus(StatusCode.ERROR, exc.getMessage());
            span.setAttribute("error.type", exc.getClass().getSimpleName());
            throw exc;
        } finally {
            span.end();
        }

        client.close();
        System.out.println("Done.");
    }
}
