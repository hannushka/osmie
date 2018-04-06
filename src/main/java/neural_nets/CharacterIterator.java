package neural_nets;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import neural_nets.spellchecker.SpellCheckIterator;
import neural_nets.anomalies.TrueFalseChIterator;

import java.nio.charset.Charset;
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

    public int totalOutcomes() {
        return inputColumns();
    }

    public static CharacterIterator getCharacterIterator(String fileLocation, String testFileLocation,
                                                         int miniBatchSize, int sequenceLength, int epochSize,
                                                          boolean minimized, boolean useCorpus) throws Exception {
        return new SpellCheckIterator(fileLocation, testFileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, epochSize, minimized, useCorpus);
    }

    public static CharacterIterator getTrueFalseIterator(String fileLocation, String testFileLocation,
                                                         int miniBatchSize, int sequenceLength, int epochSize,
                                                          boolean minimized) throws Exception {
        return new TrueFalseChIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, epochSize, minimized);
    }

    public abstract char convertIndexToCharacter(int idx);

    public abstract int getNbrClasses();

    public abstract boolean hasNextTest();

    public abstract DataSet nextTest();
}
