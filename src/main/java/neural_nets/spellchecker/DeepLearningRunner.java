package neural_nets.spellchecker;
import util.Seq2Seq;
import util.Seq2Seq.IteratorType;
import util.Seq2Seq.ScoreListener;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
            String fileLocation = "data/autoNameData.csv";
            String testFileLocation = "data/manualNameData.csv";
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("modelBRNN_quad_NO_CORPUS")
                    .useCorpus(false)
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(10000)
                    .setNbrLayers(10, 6) // params Integer int... (Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(fileLocation, testFileLocation, IteratorType.CLASSIC,false)
                    .buildNetwork()
                    .setScoreListener(ScoreListener.VISUALIZE);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
