package neural_nets.spellchecker;

import neural_nets.Seq2Seq;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
            String fileLocation = "data/shuffledTrainData.csv";
            String testFileLocation = "data/manualNameData.csv";
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("BRNN_d3_")
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(10000)
                    .setNbrLayers(50, 25) // params Integer int... (Size, Size)
                    .setLearningRate(.01)
                    .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.CLASSIC, false, true)
                    .buildNetwork()
                    //.loadModel("data/models/BRNN_240_490.bin")
                    .setScoreListener(Seq2Seq.ScoreListener.TERMINAL);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
