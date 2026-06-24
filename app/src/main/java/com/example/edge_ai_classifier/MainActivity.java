package com.example.edge_ai_classifier;

import android.os.Bundle;
import android.view.View;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;

import com.example.edge_ai_classifier.databinding.ActivityMainBinding;
import com.google.android.material.snackbar.Snackbar;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private MainViewModel       viewModel;

    private final ActivityResultLauncher<String> selectImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) viewModel.onImageSelected(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding   = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        viewModel = new ViewModelProvider(this).get(MainViewModel.class);

        setupListeners();
        observeViewModel();
    }

    private void setupListeners() {
        binding.btnSelectImage.setOnClickListener(v ->
                selectImageLauncher.launch("image/*"));
    }

    private void observeViewModel() {

        viewModel.getSelectedImageUri().observe(this, uri -> {
            if (uri != null) {
                binding.ivSelectedImage.setImageURI(uri);
                binding.ivSelectedImage.setVisibility(View.VISIBLE);
                binding.layoutPlaceholder.setVisibility(View.GONE);
            }
        });

        viewModel.getUiState().observe(this, state -> {

            if (state instanceof MainViewModel.UiState.Idle) {
                binding.progressBar.setVisibility(View.GONE);
                binding.layoutResults.setVisibility(View.GONE);
                binding.btnSelectImage.setEnabled(true);

            } else if (state instanceof MainViewModel.UiState.Loading) {
                binding.progressBar.setVisibility(View.VISIBLE);
                binding.layoutResults.setVisibility(View.GONE);
                binding.btnSelectImage.setEnabled(false);

            } else if (state instanceof MainViewModel.UiState.Success) {
                binding.progressBar.setVisibility(View.GONE);
                binding.btnSelectImage.setEnabled(true);
                showResults((MainViewModel.UiState.Success) state);

            } else if (state instanceof MainViewModel.UiState.Error) {
                MainViewModel.UiState.Error error = (MainViewModel.UiState.Error) state;
                binding.progressBar.setVisibility(View.GONE);
                binding.btnSelectImage.setEnabled(true);
                Snackbar.make(binding.getRoot(), error.message, Snackbar.LENGTH_LONG).show();
            }
        });
    }

    private void showResults(MainViewModel.UiState.Success r) {
        binding.layoutResults.setVisibility(View.VISIBLE);

        // Diagnosis
        binding.tvPrediction.setText(r.prediction);
        binding.tvConfidence.setText(r.confidence + "%");
        binding.confidenceBar.setProgress(r.confidence);
        binding.chipStatus.setVisibility(View.VISIBLE);
        binding.chipStatus.setText(r.isPositive ? "Positive" : "Normal");

        // All 4 class probabilities
        binding.tvProbCovid.setText(r.probCovid + "%");
        binding.barCovid.setProgress(r.probCovid);

        binding.tvProbNormal.setText(r.probNormal + "%");
        binding.barNormal.setProgress(r.probNormal);

        binding.tvProbPneumonia.setText(r.probPneumonia + "%");
        binding.barPneumonia.setProgress(r.probPneumonia);

        binding.tvProbTuberculosis.setText(r.probTuberculosis + "%");
        binding.barTuberculosis.setProgress(r.probTuberculosis);

        // Performance
        binding.tvInferenceTime.setText(r.inferenceTimeMs + " ms");
        binding.tvModelSize.setText(r.modelSize);

        // Memory
        binding.tvMemoryUsed.setText(r.memoryUsedMb + " MB");
        binding.tvMemoryAvailable.setText(r.memoryAvailableMb + " MB");
        binding.tvMemoryPercent.setText(r.memoryUsedPercent + "%");
        binding.memoryBar.setProgress(r.memoryUsedPercent);
    }
}