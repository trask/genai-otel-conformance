// Conformance test: Prototype memory instrumentation for AWS Bedrock.
//
// Exercises: memory operations (create_memory_store, update_memory,
// search_memory, delete_memory, delete_memory_store)
// against a mock Bedrock server, with prototype OTel spans.

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

public class AwsBedrockPrototypeMemoryTest {

    public static void main(String[] args) throws Exception {
        String mockBaseUrl = System.getenv("MOCK_LLM_URL");

        System.out.println("=== Prototype: AWS Bedrock Java Memory Conformance Test ===");

        OpenTelemetry openTelemetry = GlobalOpenTelemetry.get();
        AwsSdkTelemetry telemetry = AwsSdkTelemetry.builder(openTelemetry).build();

        AwsBedrockOtelContribTest.runMemoryOperations(mockBaseUrl, telemetry);

        System.out.println("Done.");
    }
}
