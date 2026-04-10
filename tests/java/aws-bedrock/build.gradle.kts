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
    implementation("software.amazon.awssdk:bedrockruntime:2.42.32")
    implementation("software.amazon.awssdk:apache-client:2.42.32")
    implementation("io.opentelemetry.instrumentation:opentelemetry-aws-sdk-2.2:2.26.0-alpha")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-autoconfigure:1.60.1")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-incubator:1.60.1-alpha")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.60.1")
}

val mainSourceSet = sourceSets["main"]
val configFile = rootProject.file("otel-config.yaml").absolutePath

tasks.register<JavaExec>("runOtelcontrib") {
    group = "application"
    classpath = mainSourceSet.runtimeClasspath
    mainClass.set("com.example.bedrocktest.AwsBedrockOtelContribTest")
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}

tasks.register<JavaExec>("runPrototype") {
    group = "application"
    classpath = mainSourceSet.runtimeClasspath
    mainClass.set("com.example.bedrocktest.AwsBedrockPrototypeTest")
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}
