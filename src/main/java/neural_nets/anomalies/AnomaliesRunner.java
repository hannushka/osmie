package neural_nets.anomalies;

import neural_nets.Seq2Seq;

public class AnomaliesRunner {
    public static void main(String[] args) {
        try {
            String fileLocation = "data/dataAnomalies.csv";
            String testFileLocation = "data/dataAnomaliesTest.csv";
            Seq2Seq model = AnomaliesRNN.Builder()
                    .setFilename("ARNN_")
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(5000)
                    .setLearningRate(.1)
                    .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.ANOMALIES, false)
                    .buildNetwork()
                    .setScoreListener(Seq2Seq.ScoreListener.VISUALIZE);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}