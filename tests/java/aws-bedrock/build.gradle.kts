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
    mainClass = "com.example.bedrocktest.AwsBedrockOtelContribTest"
}

dependencies {
    implementation("software.amazon.awssdk:bedrockruntime:2.42.13")
    implementation("software.amazon.awssdk:apache-client:2.42.13")
    implementation("io.opentelemetry.instrumentation:opentelemetry-aws-sdk-2.2:2.26.0-alpha")
    implementation("io.opentelemetry:opentelemetry-sdk:1.60.1")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.60.1")
}
