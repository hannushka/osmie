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
                    .setEpochSize(1000)
                    .setNbrLayers(64, 32, 16, 8, 4) // params Integer int... (Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.CLASSIC, false, false)
                    .buildNetwork()
                    //.loadModel("data/models/BRNN_240_490.bin")
                    .setScoreListener(Seq2Seq.ScoreListener.VISUALIZE);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
