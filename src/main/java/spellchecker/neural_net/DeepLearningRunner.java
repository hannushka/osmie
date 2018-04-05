package spellchecker.neural_net;
import util.Seq2Seq;
import util.Seq2Seq.IteratorType;
import util.Seq2Seq.ScoreListener;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = BiDirectionalRNN.Builder()
                    .setFilename("modelBRNN_quad_NO_CORPUS_")
                    .useCorpus(false)
                    .setBatchSize(32)
                    .setNbrEpochs(500)
                    .setEpochSize(10000)
                    //.setNbrLayers(20, 10, 6, 3) // params Integer int... (Size, Size)
                    //.setLearningRate(.1)
                    .setCharacterIterator(IteratorType.CLASSIC,false)
                    .loadModel("data/models/modelBRNN_quad_NO_CORPUS490.bin")
                    //.buildNetwork()
                    .setScoreListener(ScoreListener.VISUALIZE);
            model.runTraining();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
