package com.example.edge_ai_classifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class XRayClassifier implements AutoCloseable {

    private static final String MODEL_FILE  = "model_int8.tflite";
    private static final String LABELS_FILE = "labels.txt";
    private static final int    INPUT_SIZE  = 224;
    // 224 * 224 * 3 channels * 4 bytes per float
    private static final int    INPUT_BYTES = INPUT_SIZE * INPUT_SIZE * 3 * 4;

    private final Interpreter interpreter;
    private final List<String> labels;

    public static class Result {
        public final String  label;
        public final int     confidence;
        public final int     probNormal;
        public final int     probPneumonia;
        public final boolean isPneumonia;
        public final long    inferenceMs;

        Result(String label, int confidence, int probNormal,
               int probPneumonia, boolean isPneumonia, long inferenceMs) {
            this.label         = label;
            this.confidence    = confidence;
            this.probNormal    = probNormal;
            this.probPneumonia = probPneumonia;
            this.isPneumonia   = isPneumonia;
            this.inferenceMs   = inferenceMs;
        }
    }

    public XRayClassifier(Context context) throws IOException {
        // Load model bytes from assets into a direct ByteBuffer
        byte[] modelBytes = readAssetBytes(context, MODEL_FILE);
        ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
        modelBuffer.put(modelBytes);
        modelBuffer.rewind();

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(modelBuffer, options);

        // Load labels
        labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(context.getAssets().open(LABELS_FILE)))) {
            String line;
            while ((line = br.readLine()) != null) {
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) labels.add(trimmed);
            }
        }
    }

    public Result classify(Bitmap bitmap) {
        // 1. Resize to 224x224
        Bitmap resized = resizeBitmap(bitmap, INPUT_SIZE, INPUT_SIZE);

        // 2. Convert to float ByteBuffer, normalise pixels to [0, 1]
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_BYTES);
        inputBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int px : pixels) {
            inputBuffer.putFloat(((px >> 16) & 0xFF) / 255.0f); // R
            inputBuffer.putFloat(((px >> 8)  & 0xFF) / 255.0f); // G
            inputBuffer.putFloat(( px        & 0xFF) / 255.0f); // B
        }
        inputBuffer.rewind();

        // 3. Run inference
        float[][] output = new float[1][labels.size()];
        long startMs = System.currentTimeMillis();
        interpreter.run(inputBuffer, output);
        long inferenceMs = System.currentTimeMillis() - startMs;

        // 4. Softmax + pick winner
        float[] probs = softmax(output[0]);
        int normalIdx    = indexOfLabel("Normal");
        int pneumoniaIdx = indexOfLabel("Pneumonia");

        int probNormalPct    = normalIdx    >= 0 ? Math.round(probs[normalIdx]    * 100) : 0;
        int probPneumoniaPct = pneumoniaIdx >= 0 ? Math.round(probs[pneumoniaIdx] * 100) : 0;

        int    winnerIdx   = argmax(probs);
        String label       = winnerIdx < labels.size() ? labels.get(winnerIdx) : "Unknown";
        int    confidence  = Math.round(probs[winnerIdx] * 100);
        boolean isPneumonia = label.equalsIgnoreCase("Pneumonia");

        return new Result(label, confidence, probNormalPct,
                probPneumoniaPct, isPneumonia, inferenceMs);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static Bitmap resizeBitmap(Bitmap src, int w, int h) {
        if (src.getWidth() == w && src.getHeight() == h) return src;
        Matrix m = new Matrix();
        m.setScale((float) w / src.getWidth(), (float) h / src.getHeight());
        return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
    }

    private static byte[] readAssetBytes(Context context, String filename) throws IOException {
        try (java.io.InputStream is = context.getAssets().open(filename)) {
            byte[] buf = new byte[is.available()];
            //noinspection ResultOfMethodCallIgnored
            is.read(buf);
            return buf;
        }
    }

    private int indexOfLabel(String name) {
        for (int i = 0; i < labels.size(); i++)
            if (labels.get(i).equalsIgnoreCase(name)) return i;
        return -1;
    }

    private static float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) if (v > max) max = v;
        float sum = 0f;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) { exp[i] = (float) Math.exp(logits[i] - max); sum += exp[i]; }
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    @Override
    public void close() {
        interpreter.close();
    }
}