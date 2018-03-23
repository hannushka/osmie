package speed.neural_net;

import spellchecker.neural_net.Helper;
import spellchecker.neural_net.Seq2Seq;
import spellchecker.neural_net.Seq2Seq.IteratorType;

public class DLSpeedRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = SpeedNetwork.Builder()
                    .setBatchSize(32)
                    .setNbrEpochs(100)
                    .setNbrLayers(20) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(IteratorType.SPEED, false)
                    .buildNetwork()
                    //.loadModel(String.format("data/models/model%s.bin", 260))
                    .setScoreListener(SpeedNetwork.ScoreListener.VISUALIZE);
            model.runTraining();
//                model.runTestingOnTrain();
            model.setCharacterIterator(IteratorType.SPEED, false);
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
