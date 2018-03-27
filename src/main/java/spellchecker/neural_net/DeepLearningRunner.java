package spellchecker.neural_net;
import util.Seq2Seq;
import util.Seq2Seq.IteratorType;
import util.Seq2Seq.ScoreListener;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("modelBRNN")
                    .useCorpus(false)
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(10000)
                    .setNbrLayers(10, 6) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(IteratorType.CLASSIC,false)
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
