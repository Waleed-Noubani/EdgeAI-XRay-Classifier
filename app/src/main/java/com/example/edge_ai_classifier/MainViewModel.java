package com.example.edge_ai_classifier;

import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class MainViewModel extends ViewModel {

    // --- UiState كـ inner classes بدل sealed class ---
    public static abstract class UiState {
        public static class Idle extends UiState {}
        public static class Loading extends UiState {}
        public static class Success extends UiState {
            public final String prediction;
            public final int confidence;
            public final boolean isPositive;
            public Success(String prediction, int confidence, boolean isPositive) {
                this.prediction  = prediction;
                this.confidence  = confidence;
                this.isPositive  = isPositive;
            }
        }
        public static class Error extends UiState {
            public final String message;
            public Error(String message) { this.message = message; }
        }
    }

    private final MutableLiveData<Uri>      selectedImageUri = new MutableLiveData<>();
    private final MutableLiveData<UiState>  uiState          = new MutableLiveData<>(new UiState.Idle());

    public LiveData<Uri>      getSelectedImageUri() { return selectedImageUri; }
    public LiveData<UiState>  getUiState()          { return uiState; }

    public void onImageSelected(Uri uri) {
        selectedImageUri.setValue(uri);
        classifyImage();
    }

    private void classifyImage() {
        uiState.setValue(new UiState.Loading());

        // محاكاة تأخير المعالجة (1.5 ثانية)
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            UiState.Success result = new UiState.Success("Pneumonia", 96, true);
            uiState.setValue(result);
        }, 1500);
    }
}