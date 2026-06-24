package com.example.edge_ai_classifier;

import android.app.ActivityManager;
import android.app.Application;
import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;

import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainViewModel extends AndroidViewModel {

    public static abstract class UiState {

        public static class Idle    extends UiState {}
        public static class Loading extends UiState {}

        public static class Success extends UiState {
            public final String prediction;
            public final int    confidence;
            public final boolean isPositive;
            public final int    probPneumonia;
            public final int    probNormal;
            public final long   inferenceTimeMs;
            public final String modelSize;
            public final long   memoryUsedMb;
            public final long   memoryAvailableMb;
            public final int    memoryUsedPercent;

            public Success(String prediction, int confidence, boolean isPositive,
                           int probPneumonia, int probNormal,
                           long inferenceTimeMs, String modelSize,
                           long memoryUsedMb, long memoryAvailableMb,
                           int memoryUsedPercent) {
                this.prediction        = prediction;
                this.confidence        = confidence;
                this.isPositive        = isPositive;
                this.probPneumonia     = probPneumonia;
                this.probNormal        = probNormal;
                this.inferenceTimeMs   = inferenceTimeMs;
                this.modelSize         = modelSize;
                this.memoryUsedMb      = memoryUsedMb;
                this.memoryAvailableMb = memoryAvailableMb;
                this.memoryUsedPercent = memoryUsedPercent;
            }
        }

        public static class Error extends UiState {
            public final String message;
            public Error(String message) { this.message = message; }
        }
    }

    private final MutableLiveData<Uri>     selectedImageUri = new MutableLiveData<>();
    private final MutableLiveData<UiState> uiState          =
            new MutableLiveData<>(new UiState.Idle());

    public LiveData<Uri>     getSelectedImageUri() { return selectedImageUri; }
    public LiveData<UiState> getUiState()          { return uiState; }

    // Single-thread executor keeps inference off the main thread
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public MainViewModel(Application app) {
        super(app);
    }

    public void onImageSelected(Uri uri) {
        selectedImageUri.setValue(uri);
        classifyImage(uri);
    }

    private void classifyImage(Uri uri) {
        uiState.setValue(new UiState.Loading());

        Context context = getApplication().getApplicationContext();

        executor.execute(() -> {
            try {
                // 1. Decode Bitmap from URI
                Bitmap bitmap = uriToBitmap(context, uri);

                // 2. Run TFLite inference
                try (XRayClassifier classifier = new XRayClassifier(context)) {
                    XRayClassifier.Result result = classifier.classify(bitmap);

                    // 3. Read real device memory
                    long usedMb      = readMemoryUsedMb();
                    long availableMb = readMemoryAvailableMb(context);
                    int  memPercent  = (availableMb + usedMb) > 0
                            ? (int) (usedMb * 100L / (usedMb + availableMb))
                            : 0;

                    // 4. Model file size from assets
                    String modelSize = getModelSize(context);

                    // 5. Post success on main thread
                    UiState.Success success = new UiState.Success(
                            result.label,
                            result.confidence,
                            result.isPneumonia,
                            result.probPneumonia,
                            result.probNormal,
                            result.inferenceMs,
                            modelSize,
                            usedMb,
                            availableMb,
                            memPercent
                    );
                    uiState.postValue(success);
                }

            } catch (Exception e) {
                uiState.postValue(new UiState.Error(
                        "Classification failed: " + e.getMessage()));
            }
        });
    }


    private static Bitmap uriToBitmap(Context context, Uri uri) throws IOException {
        ContentResolver cr = context.getContentResolver();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            return ImageDecoder.decodeBitmap(
                    ImageDecoder.createSource(cr, uri),
                    (decoder, info, src) -> {
                        decoder.setMutableRequired(true);
                        decoder.setTargetColorSpace(
                                android.graphics.ColorSpace.get(
                                        android.graphics.ColorSpace.Named.SRGB));
                    });
        } else {
            //noinspection deprecation
            return MediaStore.Images.Media.getBitmap(cr, uri);
        }
    }
    private static String getModelSize(Context context) {
        try {
            long bytes = context.getAssets().openFd("model_int8.tflite").getLength();
            return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
        } catch (IOException e) {
            return "N/A";
        }
    }

    public static long readMemoryUsedMb() {
        Runtime rt        = Runtime.getRuntime();
        long usedBytes    = rt.totalMemory() - rt.freeMemory();
        return usedBytes / (1024 * 1024);
    }

    public static long readMemoryAvailableMb(Context context) {
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        ActivityManager am = (ActivityManager)
                context.getSystemService(Context.ACTIVITY_SERVICE);
        if (am != null) {
            am.getMemoryInfo(info);
            return info.availMem / (1024 * 1024);
        }
        return 0;
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        executor.shutdownNow();
    }
}