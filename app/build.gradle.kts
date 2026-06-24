plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.edge_ai_classifier"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.edge_ai_classifier"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        viewBinding = true
    }

    // Prevent compression of .tflite model files so TFLite can mmap them
    androidResources {
        noCompress += listOf("tflite")
    }
}
configurations.all {
    exclude(group = "com.google.ai.edge.litert", module = "litert")
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
}

dependencies {
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    
    implementation("androidx.lifecycle:lifecycle-viewmodel:2.8.7")
    implementation("androidx.lifecycle:lifecycle-livedata:2.8.7")

    // tensorflow dependencies
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
