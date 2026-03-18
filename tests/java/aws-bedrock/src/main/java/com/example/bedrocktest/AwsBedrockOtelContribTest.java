// Conformance test: OTel contrib opentelemetry-aws-sdk-2.2 instrumentation.
//
// Exercises: converse (Bedrock Converse API)
// against a mock Bedrock server, with the OTel AWS SDK Java library instrumentation.

package com.example.bedrocktest;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import io.opentelemetry.instrumentation.awssdk.v2_2.AwsSdkTelemetry;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.client.config.ClientOverrideConfiguration;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ConversationRole;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseRequest;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseResponse;
import software.amazon.awssdk.services.bedrockruntime.model.Message;
import software.amazon.awssdk.services.bedrockagentcore.BedrockAgentCoreClient;
import software.amazon.awssdk.services.bedrockagentcore.model.BatchCreateMemoryRecordsRequest;
import software.amazon.awssdk.services.bedrockagentcore.model.BatchCreateMemoryRecordsResponse;
import software.amazon.awssdk.services.bedrockagentcore.model.BatchDeleteMemoryRecordsRequest;
import software.amazon.awssdk.services.bedrockagentcore.model.BatchDeleteMemoryRecordsResponse;
import software.amazon.awssdk.services.bedrockagentcore.model.MemoryRecordCreateInput;
import software.amazon.awssdk.services.bedrockagentcore.model.MemoryRecordDeleteInput;
import software.amazon.awssdk.services.bedrockagentcore.model.MemoryContent;
import software.amazon.awssdk.services.bedrockagentcore.model.RetrieveMemoryRecordsRequest;
import software.amazon.awssdk.services.bedrockagentcore.model.RetrieveMemoryRecordsResponse;
import software.amazon.awssdk.services.bedrockagentcore.model.SearchCriteria;
import software.amazon.awssdk.services.bedrockagentcorecontrol.BedrockAgentCoreControlClient;
import software.amazon.awssdk.services.bedrockagentcorecontrol.model.CreateMemoryRequest;
import software.amazon.awssdk.services.bedrockagentcorecontrol.model.CreateMemoryResponse;
import software.amazon.awssdk.services.bedrockagentcorecontrol.model.DeleteMemoryRequest;

import java.net.URI;
import java.time.Instant;
import java.time.temporal.ChronoUnit;

public class AwsBedrockOtelContribTest {

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");

        System.out.println("=== OTel Contrib: AWS Bedrock Java Conformance Test ===");

        OpenTelemetry openTelemetry = GlobalOpenTelemetry.get();

        // Create AWS SDK OTel instrumentation
        AwsSdkTelemetry telemetry = AwsSdkTelemetry.builder(openTelemetry).build();

        // Create instrumented Bedrock client pointing at mock server
        BedrockRuntimeClient client = BedrockRuntimeClient.builder()
                .endpointOverride(new URI(mockBaseUrl))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .overrideConfiguration(ClientOverrideConfiguration.builder()
                        .addExecutionInterceptor(telemetry.createExecutionInterceptor())
                        .build())
                .build();

        // Run scenarios
        runConverse(client);

        client.close();
        System.out.println("Done.");
    }

    static void runConverse(BedrockRuntimeClient client) {
        System.out.println("  [converse] Bedrock Converse API");
        ConverseResponse response = client.converse(ConverseRequest.builder()
                .modelId("anthropic.claude-3-haiku-20240307-v1:0")
                .messages(Message.builder()
                        .role(ConversationRole.USER)
                        .content(ContentBlock.fromText("Say hello."))
                        .build())
                .build());
        String text = response.output().message().content().get(0).text();
        System.out.println("    -> " + text.substring(0, Math.min(60, text.length())));
    }

    /**
     * Exercise Bedrock AgentCore Memory operations for conformance testing.
     *
     * Uses the actual AWS SDK BedrockAgentCoreClient with manual OTel spans
     * to demonstrate which gen_ai.memory.* attributes are capturable.
     */
    static void runMemoryOperations(String baseUrl, AwsSdkTelemetry telemetry) throws Exception {
        URI parsedUri = new URI(baseUrl);
        String serverAddress = parsedUri.getHost();
        int serverPort = parsedUri.getPort() > 0 ? parsedUri.getPort() : 443;

        Tracer tracer = GlobalOpenTelemetry.getTracer("gen_ai.memory.aws_bedrock");

        BedrockAgentCoreClient agentcore = BedrockAgentCoreClient.builder()
                .endpointOverride(new URI(baseUrl))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .overrideConfiguration(ClientOverrideConfiguration.builder()
                        .addExecutionInterceptor(telemetry.createExecutionInterceptor())
                        .build())
                .build();

        BedrockAgentCoreControlClient control = BedrockAgentCoreControlClient.builder()
                .endpointOverride(new URI(baseUrl))
                .credentialsProvider(StaticCredentialsProvider.create(
                        AwsBasicCredentials.create("mock", "mock")))
                .region(Region.US_EAST_1)
                .overrideConfiguration(ClientOverrideConfiguration.builder()
                        .addExecutionInterceptor(telemetry.createExecutionInterceptor())
                        .build())
                .build();

        String memoryName = "conformance-test-memory-store";

        // 0. Create memory store (create_memory_store span)
        System.out.println("  [create_memory_store] Bedrock AgentCore CreateMemory");
        Span createStoreSpan = tracer.spanBuilder("create_memory_store")
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        String memoryId;
        try (Scope ignored = createStoreSpan.makeCurrent()) {
            createStoreSpan.setAttribute("gen_ai.operation.name", "create_memory_store");
            createStoreSpan.setAttribute("gen_ai.provider.name", "aws.bedrock");
            createStoreSpan.setAttribute("gen_ai.memory.store.name", memoryName);
            createStoreSpan.setAttribute("server.address", serverAddress);
            createStoreSpan.setAttribute("server.port", serverPort);
            int eventExpiryDuration = 86400;
            Instant expiration = Instant.now().plus(eventExpiryDuration, ChronoUnit.SECONDS);
            createStoreSpan.setAttribute("gen_ai.memory.expiration_date", expiration.toString());
            CreateMemoryResponse createMemoryResp = control.createMemory(
                    CreateMemoryRequest.builder().name(memoryName).eventExpiryDuration(eventExpiryDuration).build());
            memoryId = createMemoryResp.memory().id();
            createStoreSpan.setAttribute("gen_ai.memory.store.id", memoryId);
            System.out.println("    -> created memory store: " + memoryId);
        } catch (Exception e) {
            createStoreSpan.setStatus(StatusCode.ERROR, e.getMessage());
            createStoreSpan.setAttribute("error.type", e.getClass().getName());
            throw e;
        } finally {
            createStoreSpan.end();
        }

        // 1. Create memory records (update_memory span)
        System.out.println("  [update_memory] Bedrock AgentCore BatchCreateMemoryRecords");
        Instant now = Instant.now();
        String contentText = "The user prefers concise answers.";

        Span updateSpan = tracer.spanBuilder("update_memory")
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        String recordId;
        try (Scope ignored = updateSpan.makeCurrent()) {
            updateSpan.setAttribute("gen_ai.operation.name", "update_memory");
            updateSpan.setAttribute("gen_ai.provider.name", "aws.bedrock");
            updateSpan.setAttribute("gen_ai.memory.store.id", memoryId);
            updateSpan.setAttribute("gen_ai.memory.store.name", memoryName);
            updateSpan.setAttribute("gen_ai.memory.record.content", contentText);
            updateSpan.setAttribute("server.address", serverAddress);
            updateSpan.setAttribute("server.port", serverPort);
            BatchCreateMemoryRecordsResponse createResp = agentcore.batchCreateMemoryRecords(
                    BatchCreateMemoryRecordsRequest.builder()
                            .memoryId(memoryId)
                            .records(
                                    MemoryRecordCreateInput.builder()
                                            .requestIdentifier("req-001")
                                            .namespaces("conformance-test")
                                            .content(MemoryContent.builder().text(contentText).build())
                                            .timestamp(now)
                                            .build()
                            )
                            .build());
            recordId = createResp.successfulRecords().isEmpty() ? "unknown"
                    : createResp.successfulRecords().get(0).memoryRecordId();
            updateSpan.setAttribute("gen_ai.memory.record.id", recordId);
            System.out.println("    -> created record: " + recordId);
        } catch (Exception e) {
            updateSpan.setStatus(StatusCode.ERROR, e.getMessage());
            updateSpan.setAttribute("error.type", e.getClass().getName());
            throw e;
        } finally {
            updateSpan.end();
        }

        // 2. Retrieve memory records (search_memory span)
        System.out.println("  [search_memory] Bedrock AgentCore RetrieveMemoryRecords");
        String searchQuery = "What does the user prefer?";

        Span searchSpan = tracer.spanBuilder("search_memory")
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        try (Scope ignored = searchSpan.makeCurrent()) {
            searchSpan.setAttribute("gen_ai.operation.name", "search_memory");
            searchSpan.setAttribute("gen_ai.provider.name", "aws.bedrock");
            searchSpan.setAttribute("gen_ai.memory.store.id", memoryId);
            searchSpan.setAttribute("gen_ai.memory.store.name", memoryName);
            searchSpan.setAttribute("gen_ai.memory.query.text", searchQuery);
            searchSpan.setAttribute("server.address", serverAddress);
            searchSpan.setAttribute("server.port", serverPort);
            RetrieveMemoryRecordsResponse retrieveResp = agentcore.retrieveMemoryRecords(
                    RetrieveMemoryRecordsRequest.builder()
                            .memoryId(memoryId)
                            .namespace("conformance-test")
                            .searchCriteria(SearchCriteria.builder()
                                    .searchQuery(searchQuery)
                                    .build())
                            .build());
            searchSpan.setAttribute("gen_ai.memory.search.result.count",
                    retrieveResp.memoryRecordSummaries().size());
            System.out.println("    -> retrieved " + retrieveResp.memoryRecordSummaries().size() + " records");
        } catch (Exception e) {
            searchSpan.setStatus(StatusCode.ERROR, e.getMessage());
            searchSpan.setAttribute("error.type", e.getClass().getName());
            throw e;
        } finally {
            searchSpan.end();
        }

        // 3. Delete memory records (delete_memory span)
        System.out.println("  [delete_memory] Bedrock AgentCore BatchDeleteMemoryRecords");

        Span deleteSpan = tracer.spanBuilder("delete_memory")
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        try (Scope ignored = deleteSpan.makeCurrent()) {
            deleteSpan.setAttribute("gen_ai.operation.name", "delete_memory");
            deleteSpan.setAttribute("gen_ai.provider.name", "aws.bedrock");
            deleteSpan.setAttribute("gen_ai.memory.store.id", memoryId);
            deleteSpan.setAttribute("gen_ai.memory.store.name", memoryName);
            deleteSpan.setAttribute("gen_ai.memory.scope", "conformance-test");
            deleteSpan.setAttribute("gen_ai.memory.record.id", recordId);
            deleteSpan.setAttribute("server.address", serverAddress);
            deleteSpan.setAttribute("server.port", serverPort);
            BatchDeleteMemoryRecordsResponse deleteResp = agentcore.batchDeleteMemoryRecords(
                    BatchDeleteMemoryRecordsRequest.builder()
                            .memoryId(memoryId)
                            .records(MemoryRecordDeleteInput.builder()
                                    .memoryRecordId(recordId)
                                    .build())
                            .build());
            System.out.println("    -> deleted " + deleteResp.successfulRecords().size() + " records");
        } catch (Exception e) {
            deleteSpan.setStatus(StatusCode.ERROR, e.getMessage());
            deleteSpan.setAttribute("error.type", e.getClass().getName());
            throw e;
        } finally {
            deleteSpan.end();
        }

        // 4. Delete memory store (delete_memory_store span)
        System.out.println("  [delete_memory_store] Bedrock AgentCore DeleteMemory");

        Span deleteStoreSpan = tracer.spanBuilder("delete_memory_store")
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();
        try (Scope ignored = deleteStoreSpan.makeCurrent()) {
            deleteStoreSpan.setAttribute("gen_ai.operation.name", "delete_memory_store");
            deleteStoreSpan.setAttribute("gen_ai.provider.name", "aws.bedrock");
            deleteStoreSpan.setAttribute("gen_ai.memory.store.id", memoryId);
            deleteStoreSpan.setAttribute("gen_ai.memory.store.name", memoryName);
            deleteStoreSpan.setAttribute("server.address", serverAddress);
            deleteStoreSpan.setAttribute("server.port", serverPort);
            control.deleteMemory(DeleteMemoryRequest.builder()
                    .memoryId(memoryId)
                    .build());
            System.out.println("    -> deleted memory store: " + memoryId);
        } catch (Exception e) {
            deleteStoreSpan.setStatus(StatusCode.ERROR, e.getMessage());
            deleteStoreSpan.setAttribute("error.type", e.getClass().getName());
            throw e;
        } finally {
            deleteStoreSpan.end();
        }

        agentcore.close();
        control.close();
    }
}
