package spellchecker.neural_net;
import spellchecker.neural_net.Seq2Seq.IteratorType;
import spellchecker.neural_net.Seq2Seq.ScoreListener;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = RNN.Builder()
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(4000)
                    .setNbrLayers(10) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(IteratorType.TRANSLATER,false)
                    .buildNetwork()
                    //.loadModel(String.format("data/models/model%s.bin", 260))
                    .setScoreListener(ScoreListener.VISUALIZE);
            model.runTraining();
//            model.runTestingOnTrain();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
