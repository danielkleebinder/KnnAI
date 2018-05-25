package at.fhtw.ai.knn.heuristic;

import at.fhtw.ai.knn.DataSet;

/**
 * Heuristic computer for double data sets.
 *
 * @author Daniel Kleebinder
 * @since 0.1
 */
public class DoubleHeuristicComputer extends HeuristicComputer<Double, Double> {

    /**
     * Heuristic Minowski algorithm value.
     */
    private double heuristicAlgorithmValue = 2;

    /**
     * Creates a new double heuristic computer.
     */
    public DoubleHeuristicComputer() {
    }

    /**
     * Creates a new double heuristic computer using the given candidate.
     *
     * @param candidate Data set candidate.
     */
    public DoubleHeuristicComputer(DataSet<Double> candidate) {
        setCandidate(candidate);
    }

    @Override
    public void setHeuristic(Heuristic heuristic) {
        super.setHeuristic(heuristic);
        if (null == heuristic) {
            return;
        }

        switch (heuristic) {
            case Manhatten:
                heuristicAlgorithmValue = 1;
                break;
            case Euklid:
                heuristicAlgorithmValue = 2;
                break;
            default:
                throw new IllegalArgumentException("Given heuristic not supported!");
        }
    }

    @Override
    public void setCandidate(DataSet<Double> candidate) {
        super.setCandidate(candidate);
        dimensions = candidate.getDimensions();
    }

    // Runtime optimization
    int dimensions;
    double dist;
    double x1, c, weight;
    double diff;

    // Extract for optimization
    Object[] elementData, candidateElementData;

    @Override
    public float computeHeuristic(DataSet<Double> ds) {
        elementData = (Object[]) ds.elementData;
        candidateElementData = (Object[]) candidate.elementData;

        // Compute heuristic distance between the n-dimensional vectors
        weight = 1.0f;
        dist = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            if (weights != null) {
                weight = weights.get(i);
            }

            // Use API reflection hack to accelerate API
            //  x1 = ds.fastAttributeGet(i) * weight;
            //  c = candidate.fastAttributeGet(i) * weight;
            x1 = ((double) elementData[i]) * weight;
            c = ((double) candidateElementData[i]) * weight;

            // Calculate distance heuristic
            diff = x1 - c;

            // Use CPU optimized code instead of:
            //  pow(abs(x - c)), p)
            //  dist1 += Math.pow(Math.abs(x1 - c), heuristicAlgorithmValue);
            if (heuristicAlgorithmValue == 1) {
                dist += Math.abs(diff);
            } else {
                dist += diff * diff;
            }
        }

        return (float) dist;
    }
}
