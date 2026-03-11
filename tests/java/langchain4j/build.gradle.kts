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
    mainClass = "com.example.langchain4jtest.LangChain4JOtelContribTest"
}

// Force OkHttp 4.x: the OTel OTLP exporter transitively pulls in OkHttp 5.x, but
// langchain4j-open-ai depends on okhttp-sse 4.x which uses okhttp3.internal.Util
// (removed in 5.x). The OTel exporter is compatible with OkHttp 4, so pin everything
// to 4.12.0 to keep okhttp and okhttp-sse aligned.
configurations.all {
    resolutionStrategy.eachDependency {
        if (requested.group == "com.squareup.okhttp3") {
            useVersion("4.12.0")
        }
    }
}

dependencies {
    implementation("dev.langchain4j:langchain4j:0.36.2")
    implementation("dev.langchain4j:langchain4j-open-ai:0.36.2")
    implementation("io.opentelemetry:opentelemetry-sdk:1.59.0")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp:1.59.0")
}
