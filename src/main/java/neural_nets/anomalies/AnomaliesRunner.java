package neural_nets.anomalies;

import neural_nets.BiDirectionalRNN;
import neural_nets.Seq2Seq;

public class AnomaliesRunner {
    public static void main(String[] args) {
        try {
            String fileLocation = "data/dataAnomalies.csv";
            String testFileLocation = "data/dataAnomaliesTest.csv";
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("modelBRNN_")
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(10000)
                    .setNbrLayers(20, 10) // params Integer int... (Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.ANOMALIES)
                    .buildNetwork()
                    .setScoreListener(Seq2Seq.ScoreListener.VISUALIZE);
            model.runTraining();
//            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
