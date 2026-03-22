plugins {
    java
    application
}

group = "com.example"
version = "0.0.1"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}

repositories {
    mavenCentral()
}

application {
    mainClass = "com.example.bedrockagenttest.AwsBedrockAgentManualTest"
}

dependencies {
    implementation("software.amazon.awssdk:bedrockagentruntime:2.42.13")
    implementation("software.amazon.awssdk:apache-client:2.42.13")
    implementation("io.opentelemetry:opentelemetry-api:1.60.1")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-autoconfigure:1.60.1")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-incubator:1.60.1-alpha")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.60.1")
}

tasks.named<JavaExec>("run") {
    val configFile = rootProject.file("../otel-config.yaml").absolutePath
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}
