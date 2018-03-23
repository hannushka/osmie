package speed.neural_net;

import util.Seq2Seq;
import util.Seq2Seq.IteratorType;

public class DLSpeedRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = SpeedNetwork.Builder()
                    .setBatchSize(64)
                    .setEpochSize(4000)
                    .setNbrEpochs(100)
                    .setNbrLayers(20, 20) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(IteratorType.SPEED, false)
                    .buildNetwork()
                    //.loadModel(String.format("data/models/model%s.bin", 260))
                    .setScoreListener(SpeedNetwork.ScoreListener.VISUALIZE);
            model.runTraining();
//            model.runTestingOnTrain();
            model.setCharacterIterator(IteratorType.SPEED, false);
            model.runTesting(false);
            System.out.println("DONE");
            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
