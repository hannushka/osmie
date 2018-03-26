package speed.neural_net;

import util.Seq2Seq;

public class DLSpeedRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = SpeedNetwork.Builder()
                    .setBatchSize(64)
                    .setEpochSize(500)
                    .setNbrEpochs(100)
                    .setNbrLayers(10, 10) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(Seq2Seq.IteratorType.SPEED, false)
                    .buildNetwork()
                    //.loadModel(String.format("data/models/model%s.bin", 260))
                    .setScoreListener(SpeedNetwork.ScoreListener.VISUALIZE);
            model.runTraining();
//            model.runTestingOnTrain();
            System.out.println("DONE");
            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
