package neural_nets;

import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.List;

public abstract class CharacterIterator implements DataSetIterator {

    public List<String> getLabels() {
        return null;    // Returning the same as another example from DL4j that uses seq-2-seq
    }

    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public void remove() {
        throw new UnsupportedOperationException();
    }

    public boolean asyncSupported() {
        return true;
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public abstract char convertIndexToCharacter(int idx);

}
