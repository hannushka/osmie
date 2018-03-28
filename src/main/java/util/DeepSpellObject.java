package util;

public class DeepSpellObject extends SpellObject{
    double[][] distribution;

    public DeepSpellObject(double[][] distribution, String guessedWord) {
        super(-1);
        super.addName(guessedWord);
    }


}
