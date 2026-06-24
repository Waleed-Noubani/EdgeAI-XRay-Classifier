package com.example.edge_ai_classifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class XRayClassifier implements AutoCloseable {

    private static final String MODEL_FILE  = "model_int8.tflite";
    private static final String LABELS_FILE = "labels.txt";
    private static final int    INPUT_SIZE  = 224;

    private final Interpreter  interpreter;
    private final List<String> labels;
    private final int          numClasses;

    public static class Result {
        public final String  label;
        public final int     confidence;   // 0–100
        public final boolean isPneumonia;  // true for any pneumonia class
        public final float[] allProbs;     // raw probabilities for all classes
        public final long    inferenceMs;

        Result(String label, int confidence, boolean isPneumonia,
               float[] allProbs, long inferenceMs) {
            this.label       = label;
            this.confidence  = confidence;
            this.isPneumonia = isPneumonia;
            this.allProbs    = allProbs;
            this.inferenceMs = inferenceMs;
        }

        /** Probability (0–100) for a label, case-insensitive. Returns 0 if not found. */
        public int probFor(List<String> labels, String name) {
            for (int i = 0; i < labels.size(); i++)
                if (labels.get(i).equalsIgnoreCase(name))
                    return Math.round(allProbs[i] * 100);
            return 0;
        }
    }

    public XRayClassifier(Context context) throws IOException {
        byte[]     modelBytes  = readAssetBytes(context, MODEL_FILE);
        ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
        modelBuffer.put(modelBytes);
        modelBuffer.rewind();

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(modelBuffer, options);

        int[] outputShape = interpreter.getOutputTensor(0).shape(); // e.g. [1, 4]
        numClasses = outputShape[outputShape.length - 1];

        labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(context.getAssets().open(LABELS_FILE)))) {
            String line;
            while ((line = br.readLine()) != null) {
                String t = line.trim();
                if (!t.isEmpty()) labels.add(t);
            }
        }

        // Pad labels with placeholders if labels.txt has fewer entries than the model
        while (labels.size() < numClasses)
            labels.add("Class " + labels.size());
    }

    public Result classify(Bitmap bitmap) {
        Bitmap resized = resizeBitmap(bitmap, INPUT_SIZE, INPUT_SIZE);
        int inputBytes = INPUT_SIZE * INPUT_SIZE * 3 * 4; // float32
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputBytes);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int px : pixels) {
            inputBuffer.putFloat((((px >> 16) & 0xFF) / 127.5f) - 1.0f); // R
            inputBuffer.putFloat((((px >> 16) & 0xFF) / 127.5f) - 1.0f); // G
            inputBuffer.putFloat((((px >> 16) & 0xFF) / 127.5f) - 1.0f); // B
        }
        inputBuffer.rewind();

        // Output shape is [1, numClasses] — allocate exactly what the model needs
        float[][] output = new float[1][numClasses];
        long startMs = System.currentTimeMillis();
        interpreter.run(inputBuffer, output);
        long inferenceMs = System.currentTimeMillis() - startMs;

        float[] probs     = softmax(output[0]);
        int     winnerIdx = argmax(probs);
        String  label     = labels.get(winnerIdx);
        int     confidence = Math.round(probs[winnerIdx] * 100);

        // Any class that isn't "Normal" is considered a positive finding
        boolean isPneumonia = !label.equalsIgnoreCase("Normal");

        return new Result(label, confidence, isPneumonia, probs, inferenceMs);
    }

    public List<String> getLabels() { return labels; }

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
    public void close() { interpreter.close(); }
}