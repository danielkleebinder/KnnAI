package at.fhtw.ai.knn;

import at.fhtw.ai.knn.analysis.ConfusionMatrix;
import at.fhtw.ai.knn.heuristic.HeuristicComputer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The k-Nearest-Neighbor artificial intelligence AI.
 *
 * @author Daniel Kleebinder
 * @since 0.1
 */
public class KnnAI {

    /**
     * If debug outputs should be printed or not.
     */
    public static boolean debugMode = false;

    /**
     * Nobody is allowed to create an instance of the KnnAI class.
     */
    private KnnAI() {
    }

    /**
     * Predicts the quality attribute of the given <code>testData</code> set. The algorithm will use the <code>trainData</code> data set to
     * learn specific abstract concepts about the information given.
     *
     * @param <T> Generic data type.
     * @param trainData Train data set.
     * @param testData Test data set.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> trainData, List<DataSet<T>> testData, HeuristicComputer heuristicComputer) {
        return predict(trainData, testData, heuristicComputer, 10);
    }

    /**
     * Predicts the quality attribute of the given <code>testData</code> set. The algorithm will use the <code>trainData</code> data set to
     * learn specific abstract concepts about the information given.
     *
     * @param <T> Generic data type.
     * @param trainData Train data set.
     * @param testData Test data set.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @param k k-NN prediction part (default 10).
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> trainData, List<DataSet<T>> testData, HeuristicComputer heuristicComputer, int k) {
        return predict(trainData, testData, heuristicComputer, k, null);
    }

    /**
     * Predicts the quality attribute of the given <code>testData</code> set. The algorithm will use the <code>trainData</code> data set to
     * learn specific abstract concepts about the information given.
     *
     * @param <T> Generic data type.
     * @param trainData Train data set.
     * @param testData Test data set.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @param k k-NN prediction part (default 10).
     * @param weights Weights.
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> trainData, List<DataSet<T>> testData, HeuristicComputer heuristicComputer, int k, List<T> weights) {
        return predict(trainData, testData, heuristicComputer, k, weights, HeuristicComputer.Heuristic.Euklid);
    }

    /**
     * Predicts the quality attribute of the given <code>testData</code> set. The algorithm will use the <code>trainData</code> data set to
     * learn specific abstract concepts about the information given.
     *
     * @param <T> Generic data type.
     * @param trainData Train data set.
     * @param testData Test data set.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @param k k-NN prediction part (default 10).
     * @param weights Weights.
     * @param heuristic Heuristic algorithm used.
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> trainData, List<DataSet<T>> testData, HeuristicComputer heuristicComputer, int k, List<T> weights, HeuristicComputer.Heuristic heuristic) {
        heuristicComputer.setHeuristic(heuristic);

        ConfusionMatrix confusionMatrix = new ConfusionMatrix();
        confusionMatrix.setPredictionTime(System.currentTimeMillis());

        DataSet<T>[] trainDataArray = trainData.toArray(new DataSet[trainData.size()]);

        for (DataSet<T> currentTestDataSet : testData) {
            heuristicComputer.setCandidate(currentTestDataSet);
            SimplePair<DataSet<T>, Float>[] topKValues = topEntries(trainDataArray, heuristicComputer, 11);

            Object qualityAttribute;
            Map<Object, Integer> appearances = new HashMap<>(k);
            for (int i = 0; i < 11; i++) {              // Use prime number as prediction factor
                qualityAttribute = topKValues[i].key.getQualityAttribute();

                // Increase appearance counter to find the most often used quality attribute
                appearances.putIfAbsent(qualityAttribute, 1);
                if (appearances.containsKey(qualityAttribute)) {
                    appearances.put(qualityAttribute, appearances.get(qualityAttribute) + 1);
                }
            }

            // Retrieve the most often appearance entry
            Map.Entry<Object, Integer> mostOften = null;
            for (Map.Entry<Object, Integer> entry : appearances.entrySet()) {
                if (mostOften == null) {
                    mostOften = entry;
                    break;
                }

                if (entry.getValue() > mostOften.getValue()) {
                    mostOften = entry;
                }
            }

            // Set confusion matrix
            Object obj = currentTestDataSet.getQualityAttribute();
            if (mostOften != null
                    && ((obj == null && mostOften.getKey() == null)
                    || obj.equals(mostOften.getKey()))) {
                confusionMatrix.updateCorrectPredictions(1);
            } else {
                confusionMatrix.updateWrongPredictions(1);
            }

            // Update prediction matrix
            confusionMatrix.updatePredictionMatrix(obj, mostOften.getKey(), 1);
        }

        // Calculate computation time
        confusionMatrix.setPredictionTime(System.currentTimeMillis() - confusionMatrix.getPredictionTime());
        return confusionMatrix;
    }

    /**
     * Tries to correctly predict the <code>1/k</code> quality attribute of the test data part of the given data set.
     *
     * @param <T> Generic data type.
     * @param data Data.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> data, HeuristicComputer heuristicComputer) {
        return predict(data, heuristicComputer, 10);
    }

    /**
     * Tries to correctly predict the <code>1/k</code> quality attribute of the test data part of the given data set.
     *
     * @param <T> Generic data type.
     * @param data Data.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @param k k-NN prediction part (default 10).
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> data, HeuristicComputer heuristicComputer, int k) {
        return predict(data, heuristicComputer, k, null);
    }

    /**
     * Tries to correctly predict the <code>1/k</code> quality attribute of the test data part of the given data set.
     *
     * @param <T> Generic data type.
     * @param data Data.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @param k k-NN prediction part (default 10).
     * @param weights Weights.
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> data, HeuristicComputer heuristicComputer, int k, List<T> weights) {
        return predict(data, heuristicComputer, k, weights, HeuristicComputer.Heuristic.Euklid);
    }

    /**
     * Tries to correctly predict the <code>1/k</code> quality attribute of the test data part of the given data set.
     *
     * @param <T> Generic data type.
     * @param data Data.
     * @param heuristicComputer Heuristic computer for correct prediction.
     * @param k k-NN prediction part (default 10).
     * @param weights Weights.
     * @param heuristic Heuristic algorithm used.
     * @return Confision matrix for AI analysis.
     */
    public static <T> ConfusionMatrix predict(List<DataSet<T>> data, HeuristicComputer heuristicComputer, int k, List<T> weights, HeuristicComputer.Heuristic heuristic) {
        int randomIndex = 0;
        int dataSize = data.size();
        int batchSize = dataSize / k;

        // Resulting confusion matrix
        final ConfusionMatrix result = new ConfusionMatrix();
        result.setPredictionTime(System.currentTimeMillis());

        // Split data into quality groups
        Map<Object, List<DataSet<T>>> qualityAttributeBlocks = new HashMap<>();
        for (DataSet<T> current : data) {
            qualityAttributeBlocks.putIfAbsent(current.qualityAttribute, new ArrayList<>(512));
            qualityAttributeBlocks.get(current.qualityAttribute).add(current);
        }

        // Initialize data blocks
        List<List<DataSet<T>>> blocks = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            blocks.add(new ArrayList<>(batchSize));
        }

        // Retreive test and train data sets
        for (List<DataSet<T>> current : qualityAttributeBlocks.values()) {

            // Calculate batch size for current quality group
            int batchDataSize = Math.max(current.size() / k, 1);

            // Every data block needs <batchDataSize>-many entries from the quality group
            for (List<DataSet<T>> dataBlock : blocks) {

                // Use <batchDataSize>-many data sets
                for (int i = 0; i < batchDataSize; i++) {
                    if (current.isEmpty()) {
                        break;
                    }

                    // Generate a random index
                    randomIndex = (int) (Math.random() * (current.size() - 1));
                    dataBlock.add(current.get(randomIndex));

                    // Remove from list in map to prevent the same data set from occurring multiple times
                    current.remove(randomIndex);
                }
            }
        }

        // Track preparation time
        result.setPredictionTime(System.currentTimeMillis() - result.getPredictionTime());

        // Use every data block once as test data set
        List<DataSet<T>> train = new ArrayList<>(data.size());
        for (int i = 0; i < k; i++) {

            // Clear train data set
            train.clear();
            for (int j = 0; j < k; j++) {

                // Index i is used as test block
                if (i == j) {
                    continue;
                }

                // Rest is added to train data
                train.addAll(blocks.get(j));
            }

            // Predict the test data and update the confusion matrix
            ConfusionMatrix currentConfusionMatrix = predict(train, blocks.get(i), heuristicComputer, k, weights, heuristic);
            result.updatePredictionMatrix(currentConfusionMatrix);
            result.updateCorrectPredictions(currentConfusionMatrix.getNumberOfCorrectPredictions());
            result.updateWrongPredictions(currentConfusionMatrix.getNumberOfWrongPredictions());
            result.setPredictionTime(result.getPredictionTime() + currentConfusionMatrix.getPredictionTime());

            // Debug output log
            if (debugMode) {
                System.out.println("Progress: " + (((i + 1) / (double) k) * 100.0) + "%");
                System.out.println(" -> Total Time: " + result.getPredictionTime() + " ms");
                System.out.println(" -> Prediction Time: " + currentConfusionMatrix.getPredictionTime() + " ms");
                System.out.println(" -> Correct Predictions: " + currentConfusionMatrix.getNumberOfCorrectPredictions());
                System.out.println(" -> Wrong Predictions: " + currentConfusionMatrix.getNumberOfWrongPredictions());
                System.out.println();
            }
        }

        return result;
    }

    /**
     * Calculates and returns an array of the top n values in the given list.
     *
     * @param <T> Generic data type.
     * @param trainData Train data.
     * @param heuristicComputer Heuristic computer.
     * @param n Top n values.
     * @return Array containing the top n values.
     */
    private static <T> SimplePair<DataSet<T>, Float>[] topEntries(DataSet<T>[] trainData, HeuristicComputer heuristicComputer, int n) {
        SimplePair<DataSet<T>, Float>[] result = new SimplePair[n];
        SimplePair<Integer, Float> buffer = new SimplePair<>();

        // Pre initialize variables
        int currentIndex = 0;
        int biggestHeuristicIndex = 0;
        float currentHeuristic = 0;
        float biggestHeuristicInResult = Float.NEGATIVE_INFINITY;
        DataSet<T> current;

        for (int i = 0; i < trainData.length; i++) {
            current = trainData[i];
            // Compute heuristic of the current data set
            currentHeuristic = heuristicComputer.computeHeuristic(current);

            // Fill result set with n values at the beginning
            if (currentIndex < n) {
                result[currentIndex] = new SimplePair<>(current, currentHeuristic);

                // Set current biggest value in result set and corresponding index
                // for performance optimization
                if (biggestHeuristicInResult < currentHeuristic) {
                    biggestHeuristicInResult = currentHeuristic;
                    biggestHeuristicIndex = currentIndex;
                }

                currentIndex++;
                continue;
            }

            // Check if the biggest heuristic in the result set is smaller
            // than the current heuristic
            if (biggestHeuristicInResult < currentHeuristic) {
                continue;
            }

            // Replace old highest value with the new, smaller one
            result[biggestHeuristicIndex].set(current, currentHeuristic);
            buffer.set(biggestHeuristicIndex, currentHeuristic);

            // Fetch new highest value in result
            for (int j = 0; j < result.length; j++) {
                if (buffer.value < result[j].value) {
                    buffer.value = result[j].value;
                    buffer.key = j;
                }
            }
            biggestHeuristicInResult = buffer.value;
            biggestHeuristicIndex = buffer.key;
        }
        return result;
    }

    /**
     * A simple pair class with key and value.
     *
     * @author Daniel Kleebinder
     * @param <K> Key data type.
     * @param <V> Value data type.
     * @since 0.1
     */
    private static class SimplePair<K, V> {

        /**
         * Key.
         */
        public K key;
        /**
         * Value.
         */
        public V value;

        /**
         * Creates a new simple pair.
         */
        public SimplePair() {
        }

        /**
         * Creates a new simple pair with the given key and value.
         *
         * @param key Key.
         * @param value Value.
         */
        public SimplePair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        /**
         * Sets the key and value of the pair.
         *
         * @param key Key.
         * @param value Value.
         */
        public void set(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }
}
