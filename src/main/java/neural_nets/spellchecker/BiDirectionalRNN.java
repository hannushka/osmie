package neural_nets.spellchecker;

import neural_nets.Seq2Seq;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import symspell.SuggestItem;
import symspell.SymSpell;
import util.DeepSpellObject;
import util.Helper;
import util.StringUtils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class BiDirectionalRNN extends Seq2Seq {

    private int zeroEditDist, oneEditDist, twoEditDist, threeEditDist;
    private Random rng = new Random(12345);

    public static Seq2Seq Builder(){
        return new BiDirectionalRNN();
    }

    @Override
    public void runTraining() throws IOException {
//        super.runTraining();
        int miniBatchNumber = 0, generateSamplesEveryNMinibatches = 100;

        for (int i = 0; i < numEpochs; i++) {
            while (trainItr.hasNext()) {
                org.nd4j.linalg.dataset.api.DataSet ds = trainItr.next();
                net.rnnClearPreviousState();
                net.fit(ds);

                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size "
                            + miniBatchSize + " words");
                    Arrays.stream(sampleCharactersFromNetwork("A", net, (ShakeSpeareIterator) trainItr,
                            rng, 200, 3)).forEach(it -> System.out.println("==\n" + it));
                }
            }
            if(i % 5 == 0) System.out.println("Finished EPOCH #" + i);
            if(i % 10 == 0)  ModelSerializer.writeModel(net, String.format("data/models/%s%s.bin", baseFilename, i), true);
            trainItr.reset();
        }
        ModelSerializer.writeModel(net, "model.bin", true);
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
                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerDimensions[0]).activation(Activation.SOFTSIGN).build());

        for(int i = 1; i < layerDimensions.length; i++, idx++)
            builder.layer(idx, new GravesLSTM.Builder().nIn(layerDimensions[i-1]).nOut(layerDimensions[i]).activation(Activation.SOFTSIGN).build());

        for(int i = layerDimensions.length - 1; i > 0; i--, idx++)
            builder.layer(idx, new GravesLSTM.Builder().nIn(layerDimensions[i]).nOut(layerDimensions[i-1]).activation(Activation.SOFTSIGN).build());

        MultiLayerConfiguration config =  builder.layer(idx, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                .nIn(layerDimensions[0]).nOut(nOut).build())
                .build();
        net = new MultiLayerNetwork(config);
        return this;
    }

    public double runTesting(boolean print){
        Evaluation eval = new Evaluation(testItr.totalOutcomes());

        while(testItr.hasNext()){
            DataSet ds = testItr.next();
            net.rnnClearPreviousState();
            INDArray output = net.output(ds.getFeatures(), false, ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
            eval.evalTimeSeries(ds.getLabels(), output, ds.getLabelsMaskArray());
        }

        printStats();
        System.out.println(StringUtils.reduceEvalStats(eval.stats()));

        return eval.f1();
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
        String inp, out, label;
        for(int i = 0; i < inputStr.length; i++){
            inp = inputStr[i];
            out = resultStr[i];
            label = labelStr[i];

            if(print) System.out.println(inp + ",,," + out + ",,," + label);
            if(StringUtils.oneEditDist(inp, label)) editDistOne++;
            if(inp.equals(out) && inp.equals(label)) noChangeCorrect++;
            if(inp.equals(out) && !inp.equals(label)) noChangeIncorrect++;
            if(!inp.equals(label) && out.equals(label)) changedCorrectly++;
            if(inp.equals(label) && !inp.equals(out)) changedIncorrectly++;
            if(!inp.equals(label) && !inp.equals(out) && !out.equals(label))  wrongChangeType++;

            switch(StringUtils.editDist(inp, label)) {
                case 0: zeroEditDist++;
                    break;
                case 1: oneEditDist++;
                    break;
                case 2: twoEditDist++;
                    break;
                case 3:
                    threeEditDist++;
                default:
            }
        }
    }

    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        ShakeSpeareIterator iter, Random rng, int charactersToSample, int numSamples ){
        //Set up initialization. If no initialization: use a random character
        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        Map<Character, Integer> characterIntegerMap = iter.charToIdxMap;
        for( int i=0; i<init.length; i++ ){
            int idx = characterIntegerMap.get(init[i]);
            for( int j=0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

        for( int i=0; i<charactersToSample; i++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for( int s=0; s<numSamples; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
        return out;
    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }

}
