package at.fhtw.ai.knn;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A data set is the basic class for a single data unit.
 *
 * @author Daniel Kleebinder
 * @param <T> Generic data attribute type.
 * @since 0.1
 */
public class DataSet<T> {

    /**
     * Contains all attributes of the data unit.
     */
    protected final List<T> attributes = new ArrayList<>(17);
    /**
     * Array list element data.
     */
    public T[] elementData;

    /**
     * Contains the quality attribute
     */
    protected Object qualityAttribute;

    /**
     * Creates a new instance of data set.
     */
    public DataSet() {
        loadElementData();
    }

    /**
     * Loads the fast array access using reflection.
     */
    private void loadElementData() {
        try {
            Field field = attributes.getClass().getDeclaredField("elementData");
            field.setAccessible(true);
            elementData = (T[]) field.get(attributes);
        } catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Returns all attributes.
     *
     * @return Attributes.
     */
    public List<T> getAttributes() {
        return attributes;
    }

    /**
     * Fast access for the attributes array. Used internally to accelerate the AI API.
     *
     * @param index Index.
     * @return Value.
     */
    public T fastAttributeGet(int index) {
        return elementData[index];
    }

    /**
     * Returns the dimensions of the data set.
     *
     * @return Dimensions.
     */
    public int getDimensions() {
        return attributes.size();
    }

    /**
     * Sets the quality attribute which will be predicted by the KNN algorithm.
     *
     * @param qualityAttribute Quality attribute.
     */
    public void setQualityAttribute(Object qualityAttribute) {
        this.qualityAttribute = qualityAttribute;
    }

    /**
     * Returns the quality attribute.
     *
     * @return Quality attribute.
     */
    public Object getQualityAttribute() {
        return qualityAttribute;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < attributes.size(); i++) {
            result.append(attributes.get(i));
            if (i != (attributes.size() - 1)) {
                result.append(", ");
            }
        }
        return result.toString();
    }
}
