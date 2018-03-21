package speed.neural_net;

public class DLSpeedRunner {
    public static void main(String[] args) {
        try {
            SpeedNetwork model = SpeedNetwork.Builder()
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setNbrLayers(6) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(SpeedNetwork.IteratorType.TRANSLATER, 4000, true)
                    .buildBiNetwork()
                    //.loadModel(String.format("data/models/model%s.bin", 260))
                    .setScoreListener(SpeedNetwork.ScoreListener.VISUALIZE);

            model.runTraining();
//                model.runTestingOnTrain();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
