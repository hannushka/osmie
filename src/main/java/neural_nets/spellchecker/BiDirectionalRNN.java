package neural_nets.spellchecker;

import neural_nets.Seq2Seq;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import symspell.SuggestItem;
import symspell.SymSpell;
import util.DeepSpellObject;
import util.Helper;
import util.StringUtils;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BiDirectionalRNN extends Seq2Seq {
    public static Seq2Seq Builder(){
        return new BiDirectionalRNN();
    }

    @Override
    public Seq2Seq buildNetwork() throws Exception {
        int nOut = trainItr.totalOutcomes(), idx = 1, nIn = trainItr.inputColumns();
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learningRate)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
//                .regularization(true).dropOut(0.2)
                .list()
                .layer(0, new GravesBidirectionalLSTM.Builder().nIn(nIn).nOut(layerDimensions[0]).activation(Activation.SOFTSIGN).build());

        for(int i = 1; i < layerDimensions.length; i++, idx++)
            builder.layer(idx, new GravesBidirectionalLSTM.Builder().nIn(layerDimensions[i-1]).nOut(layerDimensions[i]).activation(Activation.SOFTSIGN).build());

        for(int i = layerDimensions.length - 1; i > 0; i--, idx++)
            builder.layer(idx, new GravesBidirectionalLSTM.Builder().nIn(layerDimensions[i]).nOut(layerDimensions[i-1]).activation(Activation.SOFTSIGN).build());

        MultiLayerConfiguration config =  builder.layer(idx, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                .nIn(layerDimensions[0]).nOut(nOut).build())
                .build();
        net = new MultiLayerNetwork(config);
        return this;
    }

    public double runTesting(boolean print){
        Evaluation eval = new Evaluation(testItr.totalOutcomes());
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
        return eval.f1();
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
    public void printSuggestion(String input, SymSpell symSpell) {
        DataSet ds = ((SpellCheckIterator) trainItr).createDataSetForString(input);
        net.rnnClearPreviousState();
        INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
        String[] words = Helper.convertTensorsToWords(output, trainItr);
        System.out.println("System: " + words[0]);
        List<SuggestItem> spellSugg = symSpell.lookupSpecialized(input, SymSpell.Verbosity.Closest);
        if(!spellSugg.isEmpty()) System.out.println("SymSpell: " + spellSugg.get(0).term);
        spellSugg = symSpell.lookupSpecialized(words[0], SymSpell.Verbosity.Closest);
        if(!spellSugg.isEmpty()) System.out.println("System + SymSpell: " + spellSugg.get(0).term);
    }

    public void createReadableStatistics(INDArray input, INDArray result, INDArray labels, boolean print,
                                         List<DeepSpellObject> deepSpellObjects, SymSpell symSpell){
        String[] inputStr = Helper.convertTensorsToWords(input, trainItr);
        String[] resultStr = Helper.convertTensorsToWords(result, trainItr);
        String[] labelStr = Helper.convertTensorsToWords(labels, trainItr);
/**        for(DeepSpellObject obj : deepSpellObjects){
            String word = obj.currentName.orElse("");
            if(word.contains(" ")) continue;
            List<SuggestItem> items = symSpell.lookupSpecialized(word, SymSpell.Verbosity.Closest);
            items.addAll(symSpell.lookupSpecialized(obj.inputName, SymSpell.Verbosity.Closest));
            Collections.sort(items);
            if(!items.isEmpty() && items.get(0).distance <= 1) {
                resultStr[obj.index] = items.get(0).term;
            }
        }**/
        String inp, out, label;
        Pattern pattern = Pattern.compile("s+");
        Matcher m, m2;
        ArrayList<String> matches, matches2;

        boolean equals;
        for(int i = 0; i < inputStr.length; i++){
            inp = inputStr[i];
            out = resultStr[i];
            label = labelStr[i];

            matches = new ArrayList<>();
            matches2 = new ArrayList<>();
            m = pattern.matcher(out);
            m2 = pattern.matcher(label);

            while(m.find()){
                matches.add(m.group());
            }
            while(m2.find()){
                matches2.add(m2.group());
            }

            for(int j = 0; j < Math.min(matches.size(), matches2.size()); j++){
                equals = matches.get(j).equals(matches2.get(j));
                if(equals && matches.get(j).equals("s")) sCorr++;
                else if(!equals && matches.get(j).equals("s")) sInc++;
                else if(matches.get(j).equals(matches2.get(j))) ssCorr++;
                else ssInc++;
            }

            if(print) System.out.println(inp + ",,," + out + ",,," + label);
            if(StringUtils.oneEditDist(inp, label)) editDistOne++;
            if(inp.equals(out) && inp.equals(label)) noChangeCorrect++;
            if(inp.equals(out) && !inp.equals(label)) noChangeIncorrect++;
            if(!inp.equals(label) && out.equals(label)) changedCorrectly++;
            if(inp.equals(label) && !inp.equals(out)) changedIncorrectly++;
            if(!inp.equals(label) && !inp.equals(out) && !out.equals(label))  wrongChangeType++;
        }
    }

}
