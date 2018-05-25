package ai;

import at.fhtw.ai.knn.DataSet;
import at.fhtw.ai.knn.KnnAI;
import at.fhtw.ai.knn.analysis.ConfusionMatrix;
import at.fhtw.ai.knn.heuristic.FloatHeuristicComputer;

import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test main class for white wine knn AI.
 *
 * @author Daniel Kleebinder
 */
public class WhiteWineTest {

    private static final boolean SKIP_FIRST_LINE = true;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Path dataFile = Paths.get("./data/wine/winequality-white.csv");
//        Path dataFile = Paths.get("./data/wine/winequality-white-10k.csv");
//        Path dataFile = Paths.get("./data/iris/iris_new.txt");
//        Path dataFile = Paths.get("./data/iris/iris_new_large.txt");
        byte[] bytes = null;
        try {
            bytes = Files.readAllBytes(dataFile);
        } catch (IOException ex) {
            Logger.getLogger(WhiteWineTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        String[] lines = new String(bytes).split("\n");

        String[] entries;
        List<DataSet<Float>> data = new ArrayList<>(2048);
        boolean first = true;
        for (String current : lines) {
            if (SKIP_FIRST_LINE && first) {
                first = false;
                continue;
            }
            entries = current.split(";");

            // Skip last element. The last element is used as quality indicator
            DataSet<Float> dds = new DataSet<>();
            for (int i = 0; i < entries.length - 1; i++) {
                String entry = entries[i];
                entry = entry.replaceAll(",", ".");
                dds.getAttributes().add(Float.parseFloat(entry));
            }
            dds.setQualityAttribute(entries[entries.length - 1]);
            data.add(dds);
        }

        // Run k-NN AI algorithm
        KnnAI.debugMode = false;
        ConfusionMatrix confusionMatrix = KnnAI.predict(data, new FloatHeuristicComputer(), 10);
        System.out.println("Correct Predictions: " + confusionMatrix.getNumberOfCorrectPredictions());
        System.out.println("Wrong Predictions: " + confusionMatrix.getNumberOfWrongPredictions());
        System.out.println("AI Accuracy: " + confusionMatrix.accuracy());
        System.out.println("Prediction Time: " + confusionMatrix.getPredictionTime() + " ms");
        System.out.println();
    }
}
