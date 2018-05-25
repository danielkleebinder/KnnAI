package at.fhtw.ai.knn.analysis;

import java.util.HashMap;
import java.util.Map;

/**
 * A multiclass confusion matrix implementation for the prediction and calculation of AI accuracy.
 *
 * @author Daniel Kleebinder
 * @since 0.1
 */
public class ConfusionMatrix {

    private int correctPredictions = 0;
    private int wrongPredictions = 0;

    private long predictionTime = 0;

    private Map<Object, Map<Object, Integer>> matrix = new HashMap<>();

    /**
     * Updates the prediction matrix.
     *
     * @param x X object identifier.
     * @param y Y object identifier.
     * @param delta Delta.
     */
    public void updatePredictionMatrix(Object x, Object y, int delta) {
        matrix.putIfAbsent(x, new HashMap<>());
        matrix.get(x).putIfAbsent(y, 0);

        matrix.get(x).put(y, matrix.get(x).get(y) + delta);
    }

    /**
     * Updates this prediction matrix using the given one. The values of the prediction matrix from the other confusion matrix will be added
     * to the values of the prediction matrix of this confusion matrix.
     *
     * @param other Other confusion matrix.
     */
    public void updatePredictionMatrix(ConfusionMatrix other) {
        if (other == null) {
            throw new NullPointerException("The other confusion matrix is null!");
        }

        for (Map.Entry<Object, Map<Object, Integer>> row : other.matrix.entrySet()) {
            if (!matrix.containsKey(row.getKey())) {
                matrix.put(row.getKey(), new HashMap<>(row.getValue()));
                continue;
            }
            for (Map.Entry<Object, Integer> value : row.getValue().entrySet()) {
                if (matrix.get(row.getKey()).containsKey(value.getKey())) {
                    matrix.get(row.getKey()).put(
                            value.getKey(),
                            matrix.get(row.getKey()).get(value.getKey()) + value.getValue()
                    );
                }
                matrix.get(row.getKey()).putIfAbsent(value.getKey(), value.getValue());
            }
        }
    }

    /**
     * Computes and returns the true confusion matrix as two dimensional object array.
     *
     * @return True confusion matrix.
     */
    public Object[][] confusionMatrix() {
        Object[][] result = new Object[matrix.size() + 1][matrix.size() + 1];

        Object[] a = matrix.keySet().toArray(new Object[matrix.size()]);

        int x = 0;
        for (Object key : matrix.keySet()) {
            int y = 0;
            for (Object key2 : matrix.keySet()) {
                if (y == 0) {
                    result[x + 1][y] = key;
                }
                if (x == 0) {
                    result[x][y + 1] = key2;
                }
                if (x != 0 && y != 0) {
                    result[x][y] = matrix.get(key).get(key2);
                }
                y++;
            }
            x++;
        }
        return result;
    }

    /**
     * Updates the correct predictions.
     *
     * @param delta Delta value.
     */
    public void updateCorrectPredictions(int delta) {
        correctPredictions += delta;
    }

    /**
     * Updates the wrong predictions.
     *
     * @param delta Delta value.
     */
    public void updateWrongPredictions(int delta) {
        wrongPredictions += delta;
    }

    /**
     * Returns the number of total predictions of the AI.
     *
     * @return Total predictions.
     */
    public int getNumberOfTotalPredictions() {
        return getNumberOfCorrectPredictions() + getNumberOfWrongPredictions();
    }

    /**
     * Returns the number of correct predictions of the AI.
     *
     * @return Correct predictions.
     */
    public int getNumberOfCorrectPredictions() {
        return correctPredictions;
    }

    /**
     * Returns the number of wrong predictions of the AI.
     *
     * @return Wrong predictions.
     */
    public int getNumberOfWrongPredictions() {
        return wrongPredictions;
    }

    /**
     * Sets the prediction time in milliseconds.
     *
     * @param predictionTime Prediction time.
     */
    public void setPredictionTime(long predictionTime) {
        this.predictionTime = predictionTime;
    }

    /**
     * Returns the prediction time in milliseconds.
     *
     * @return Prediction time.
     */
    public long getPredictionTime() {
        return predictionTime;
    }

    /**
     * Computes the total accuracy of the AI. The accuracy is a floating point value between 0.0 and 1.0. An accuracy of 1 is an AI with
     * 100% correct predictability.
     *
     * @return Accuracy [0;1].
     */
    public double accuracy() {
        return correctPredictions / (double) (correctPredictions + wrongPredictions);
    }
}
