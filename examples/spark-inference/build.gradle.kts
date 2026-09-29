// Plain Java. Spark is a dependency of this example only, never of a Tlaloc module.
plugins {
    java
    application
}

java {
    toolchain { languageVersion = JavaLanguageVersion.of(25) }
}

dependencies {
    implementation("io.github.pedronahum:tlaloc-runtime-pjrt:0.1.0-alpha02")
    implementation("io.github.pedronahum:tlaloc-tokenizer:0.1.0-alpha02")
    // Apache-2.0. Local mode: the driver and the executor are this JVM.
    implementation("org.apache.spark:spark-sql_2.13:4.2.0")
}

application {
    mainClass.set("SparkInference")
    applicationDefaultJvmArgs = listOf(
        "--enable-native-access=ALL-UNNAMED", "-Xmx8g",
        // Spark's own requirement on JDK 17+ (its launcher scripts pass these).
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    )
}
