import org.gradle.api.tasks.JavaExec

plugins {
    java
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

dependencies {
    implementation("software.amazon.awssdk:bedrockagentruntime:2.42.33")
    implementation("software.amazon.awssdk:apache-client:2.42.33")
    implementation("io.opentelemetry:opentelemetry-api:1.61.0")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-autoconfigure:1.61.0")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-incubator:1.60.1-alpha")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.61.0")
}

val mainSourceSet = sourceSets["main"]
val configFile = rootProject.file("otel-config.yaml").absolutePath

tasks.register<JavaExec>("runPrototype") {
    group = "application"
    classpath = mainSourceSet.runtimeClasspath
    mainClass.set("com.example.bedrockagenttest.AwsBedrockAgentPrototypeTest")
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}
