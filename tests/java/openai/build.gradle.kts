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
    implementation("com.openai:openai-java:4.26.0")
    implementation("io.opentelemetry.instrumentation:opentelemetry-openai-java-1.1:2.25.0-alpha")
    implementation("io.opentelemetry:opentelemetry-sdk:1.59.0")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.59.0")
}
