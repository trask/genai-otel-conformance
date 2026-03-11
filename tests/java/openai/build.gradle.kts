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
    mainClass = "com.example.openaitest.OpenAiOtelContribTest"
}

dependencies {
    implementation("com.openai:openai-java:4.28.0")
    implementation("io.opentelemetry.instrumentation:opentelemetry-openai-java-1.1:2.26.0-alpha")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-autoconfigure:1.60.1")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-incubator:1.60.1-alpha")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.60.1")
}

tasks.named<JavaExec>("run") {
    val configFile = rootProject.file("../otel-config.yaml").absolutePath
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}
