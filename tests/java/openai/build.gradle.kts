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
    implementation("com.openai:openai-java:4.32.0")
    implementation("io.opentelemetry.instrumentation:opentelemetry-openai-java-1.1:2.26.0-alpha")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-autoconfigure:1.61.0")
    implementation("io.opentelemetry:opentelemetry-sdk-extension-incubator:1.61.0-alpha")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.61.0")
}

val mainSourceSet = sourceSets["main"]
val configFile = rootProject.file("otel-config.yaml").absolutePath

tasks.register<JavaExec>("runOtelcontrib") {
    group = "application"
    classpath = mainSourceSet.runtimeClasspath
    mainClass.set("com.example.openaitest.OpenAiOtelContribTest")
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}

tasks.register<JavaExec>("runPrototype") {
    group = "application"
    classpath = mainSourceSet.runtimeClasspath
    mainClass.set("com.example.openaitest.OpenAiPrototypeTest")
    jvmArgs("-Dotel.config.file=$configFile", "-Dotel.java.global-autoconfigure.enabled=true")
}
