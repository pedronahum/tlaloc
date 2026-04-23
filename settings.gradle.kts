rootProject.name = "tlaloc"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

include(":core")
include(":ir")
include(":autograd")
include(":stablehlo")
include(":compiler-plugin")
