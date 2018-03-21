package spellchecker.neural_net;

public class DeepLearningRunner {
    public static void main(String[] args) {
        try {
            Seq2Seq model = EmbeddingRNN.Builder()
                    .setBatchSize(32)
                    .setNbrEpochs(10000)
                    .setNbrLayers(6, 4) // params Integer int... (#lager, Size, Size)
                    .setLearningRate(.1)
                    .setCharacterIterator(Seq2Seq.IteratorType.EMBEDDING,true)
                    .buildNetwork()
                    //.loadModel(String.format("data/models/model%s.bin", 260))
                    .setScoreListener(Seq2Seq.ScoreListener.VISUALIZE);
            model.runTraining();
//            model.runTestingOnTrain();
            model.runTesting(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
