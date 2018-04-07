package neural_nets;

import symspell.SymSpell;
import symspell.SuggestItem;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import util.*;
import util.StringUtils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class RNN extends Seq2Seq {

    public static Seq2Seq Builder(){
        return new RNN();
    }

    public void runTraining() throws IOException {
        int miniBatchNumber = 0, generateSamplesEveryNMinibatches = 100;
        for (int i = 0; i < numEpochs; i++) {
            while (trainItr.hasNext()) {
                DataSet ds = trainItr.next();
                net.rnnClearPreviousState();
                net.fit(ds);

                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size "
                            + miniBatchSize + " words");
                }
            }
            if(i % 5 == 0) System.out.println("Finished EPOCH #" + i);
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/models/%s%s.bin", baseFilename, i), true);
            trainItr.reset();
        }
        ModelSerializer.writeModel(net, "model.bin", true);
    }

    public void runTesting(boolean print){
        Evaluation eval = new Evaluation(testItr.getNbrClasses());
        List<DeepSpellObject> spellObjects = new ArrayList<>();
        SymSpell symSpell = new SymSpell(-1, 2, -1, 10);
        if(!symSpell.loadDictionary("data/korpus_freq_dict.txt", 0, 1)) try {
            throw new FileNotFoundException();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        while(testItr.hasNext()){
            DataSet ds = testItr.next();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
            List<DeepSpellObject> objects = Helper.getSpellObjectsFromUncertainTensors(ds.getFeatureMatrix(), output, ds.getLabels(), testItr);
            spellObjects.addAll(objects);

            createReadableStatistics(ds.getFeatures(), output, ds.getLabels(), print, objects, symSpell);
        }

        printStats();
        int correct = 0, destroy = 0;
        for(DeepSpellObject obj : spellObjects){
            if(obj.guessCorrect()) correct++;
            if(!obj.guessCorrect() && obj.inputName.equals(obj.correctName)) destroy++;
        }
        System.out.println(correct + " / " + spellObjects.size() + " (unsure guesses that are correct & "
                + destroy + " good->bad)");
        System.out.println(StringUtils.reduceEvalStats(eval.stats()));


        int j = 0, i = 0, k = 0, l=0;
        for(DeepSpellObject obj : spellObjects){
            String word = obj.currentName.orElse("");
            if(word.contains(" ")) continue;
            List<SuggestItem> items = symSpell.lookupSpecialized(word, SymSpell.Verbosity.Closest);
            items.addAll(symSpell.lookupSpecialized(obj.inputName, SymSpell.Verbosity.Closest));
            Collections.sort(items);
            if(!items.isEmpty() && items.get(0).distance <= 1) {
                if(items.get(0).term.equals(obj.correctName)) j++;
                if(obj.correctName.equals(word)) k++;
                if(obj.correctName.equals(word) && !items.get(0).term.equals(obj.correctName)) l++;
                i++;
            }
        }
        System.out.println("Symspell introduces " + (j-k) + " corrections extra from the unsure ones.");
        System.out.println(j + " corrections out of " + i + " where " + k + " already correct, " + l + " ruined (symspell)");

//        for(DeepSpellObject obj: spellObjects){
//            obj.generateNewWordsFromGuess();
//            DataSet ds = itr.createDataSetFromDSO(obj);
//            net.rnnClearPreviousState();
//            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
//            double[][] distr = Helper.getBestGuess(output);
//            System.out.println(Helper.getWordFromDistr(distr, itr));
//            System.out.println(obj.currentName.orElse("") + ",,," + obj.correctName);
//        }
//        eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception{
        int tbpttLength = 50, idx = 1, nOut = trainItr.totalOutcomes(), nIn = trainItr.inputColumns();
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true).l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerDimensions[0])
                        .activation(Activation.SOFTSIGN).build());

        for(int i = 1; i < layerDimensions.length; i++, idx++)
            builder.layer(idx, new GravesLSTM.Builder().nIn(layerDimensions[i-1]).nOut(layerDimensions[i]).activation(Activation.SOFTSIGN).build());  //10->5,5->2

        for(int i = layerDimensions.length - 1; i > 0; i--, idx++)
            builder.layer(idx, new GravesLSTM.Builder().nIn(layerDimensions[i]).nOut(layerDimensions[i-1]).activation(Activation.SOFTSIGN).build());

        MultiLayerConfiguration config = builder.layer(idx, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                .nIn(layerDimensions[0]).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();
        net = new MultiLayerNetwork(config);
        return this;

    }
}
