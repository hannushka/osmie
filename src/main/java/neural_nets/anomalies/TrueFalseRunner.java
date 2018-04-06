package neural_nets.anomalies;

import neural_nets.spellchecker.BiDirectionalRNN;
import util.Seq2Seq;

public class TrueFalseRunner {
    public static void main(String[] args) {
        try {
            String fileLocation = "data/autoNameData.csv";
            String testFileLocation = "";
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("modelBRNN_quad_NO_CORPUS")
                    .useCorpus(false)
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(10000)
                    .setNbrLayers(20, 10) // params Integer int... (Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.TRUEFALSE,false)
                    .buildNetwork()
                    .setScoreListener(Seq2Seq.ScoreListener.VISUALIZE);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}