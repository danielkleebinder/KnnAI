package at.fhtw.ai.knn.heuristic;

import at.fhtw.ai.knn.DataSet;

/**
 * Heuristic computer for float data sets.
 *
 * @author Daniel Kleebinder
 * @since 0.1
 */
public class FloatHeuristicComputer extends HeuristicComputer<Float, Float> {

    /**
     * Heuristic Minowski algorithm value.
     */
    private float heuristicAlgorithmValue = 2;

    /**
     * Creates a new float heuristic computer.
     */
    public FloatHeuristicComputer() {
    }

    /**
     * Creates a new float heuristic computer using the given candidate.
     *
     * @param candidate Data set candidate.
     */
    public FloatHeuristicComputer(DataSet<Float> candidate) {
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
    public void setCandidate(DataSet<Float> candidate) {
        super.setCandidate(candidate);
        dimensions = candidate.getDimensions();
    }

    // Runtime optimization
    int dimensions;
    float dist;
    float x1, c, weight;
    float diff;

    // Extract for optimization
    Object[] elementData, candidateElementData;

    @Override
    public float computeHeuristic(DataSet<Float> ds) {
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
            x1 = ((float) elementData[i]) * weight;
            c = ((float) candidateElementData[i]) * weight;

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

        return dist;
    }
}
