package at.fhtw.ai.knn.heuristic;

import at.fhtw.ai.knn.DataSet;
import java.util.List;

/**
 * Computer for heuristics.
 *
 * @author Daniel Kleebinder
 * @param <T> Generic data set type.
 * @param <Q> Weight attribute data type.
 * @since 0.1
 */
public abstract class HeuristicComputer<T, Q> {

    /**
     * Contains the available heuristic calculation methods.
     *
     * @author Daniel Kleebinder
     * @since 0.1
     */
    public static enum Heuristic {
        Euklid,
        Manhatten
    }

    /**
     * The heuristic algorithm used to compare data sets.
     */
    protected Heuristic heuristic = Heuristic.Euklid;

    /**
     * Target candidate.
     */
    protected DataSet<T> candidate;

    /**
     * Contains all weights for the attributes.
     */
    protected List<Q> weights;

    /**
     * Sets the heuristic algorithm.
     *
     * @param heuristic Heuristic.
     */
    public void setHeuristic(Heuristic heuristic) {
        this.heuristic = heuristic;
    }

    /**
     * Returns the heuristic algorithm.
     *
     * @return Heuristic.
     */
    public Heuristic getHeuristic() {
        return heuristic;
    }

    /**
     * Sets the candidate for the computer.
     *
     * @param candidate Target candidate.
     */
    public void setCandidate(DataSet<T> candidate) {
        this.candidate = candidate;
    }

    /**
     * Returns the target candidate.
     *
     * @return Candidate.
     */
    public DataSet<T> getCandidate() {
        return candidate;
    }

    /**
     * Sets the weights for the heuristic computer.
     *
     * @param weights Weights.
     */
    public void setWeights(List<Q> weights) {
        this.weights = weights;
    }

    /**
     * Returns all weights.
     *
     * @return Weights.
     */
    public List<Q> getWeights() {
        return weights;
    }

    /**
     * Computes the heuristic of the given data set to the current candidate.
     *
     * @param ds1 Data set.
     * @return Heuristic.
     */
    public abstract float computeHeuristic(DataSet<T> ds1);
}
