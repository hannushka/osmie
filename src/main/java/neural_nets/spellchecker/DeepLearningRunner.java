package neural_nets.spellchecker;

import neural_nets.Seq2Seq;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
//            String fileLocation = "data/shuffledTrainData.csv";
//            String testFileLocation = "data/manualNameData.csv";
            String fileLocation = "data/et_autoNameData.csv.merged";
            String testFileLocation = "data/et_autoNameData.csv.test";
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("BRNN_d3_et_2_")
                    .setBatchSize(32)
                    .setNbrEpochs(2000)
                    .setEpochSize(10000)
                    .setNbrLayers(20, 10, 6) // params Integer int... (Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(fileLocation, testFileLocation, Seq2Seq.IteratorType.CLASSIC,
                            false, false)
//                    .buildNetwork()
                    .loadModel("data/models/BRNN_d3_et_200.bin")
                    .setScoreListener(Seq2Seq.ScoreListener.VISUALIZE);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
