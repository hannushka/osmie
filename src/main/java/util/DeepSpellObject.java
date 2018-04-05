package util;

// TODO HARDCODED shift
public class DeepSpellObject extends SpellObject{
    static final String dkAlphabet = "data/dk_alphabet.txt";    // Singleton?
    private double[][] distribution;
    private Alphabet alphabet;
    public String correctName;

    public DeepSpellObject(double[][] distribution, String guessedWord, String correctWord) {
        super(-1);
        super.addName(guessedWord);
        this.distribution = distribution;
        this.correctName = correctWord;
        this.alphabet = Alphabet.getInstance();
    }

    public boolean guessCorrect(){
        return currentName.orElse("").equals(correctName);
    }

    /**
     * Insert: In front & behind
     * Delete: Char
     * Transpose: Backwards & Forward
     * Replace: Only with common or all?
     * ...We need DK_alphabet xD
     * Retrieve all unsure.
     * Edit them one by one & let it correct itself recursively if needed be in a new DeepSpellObject
     */

    public void generateNewWordsFromGuess(){
        double maxForTimeSeries;
        for(int i = 0; i < distribution.length; i++){
            maxForTimeSeries = Helper.getMaxDbl(distribution[i]);
            if(maxForTimeSeries < 0.5 && maxForTimeSeries > 0){
                generateEditsFromIndex(i);
            }
        }
        generateDeleteEdits();
    }

    private void generateDeleteEdits(){
        String suggestedWord = currentName.orElse("");
        for(int i = 0; i< suggestedWord.length(); i++){
            addSuggestion(new StringBuilder(suggestedWord).deleteCharAt(i).toString());
        }
    }

    private void generateEditsFromIndex(int charIndex){
        String suggestedWord = currentName.orElse("");
        StringBuilder sb = new StringBuilder(suggestedWord);
        charIndex = charIndex + 1;
        for(char c : alphabet.alphabetChars){   // Case: Replace
            sb.setCharAt(charIndex, c);
            addSuggestion(sb.toString());
        }
        sb.deleteCharAt(charIndex);             // Case: Delete
        addSuggestion(sb.toString());
        for(char c : alphabet.alphabetChars){   // Case: Insert
            if(charIndex != 0)
                addSuggestion(suggestedWord.substring(0, charIndex)
                        + c + suggestedWord.substring(charIndex));  // Before
            if(charIndex < suggestedWord.length()-1)
                addSuggestion(suggestedWord.substring(0, charIndex+1) + c + suggestedWord.substring(charIndex+1));  // Behind
        }

        // Case: Transpose
        sb = new StringBuilder(suggestedWord);
        sb.setCharAt(charIndex-1, suggestedWord.charAt(charIndex));
        sb.setCharAt(charIndex, suggestedWord.charAt(charIndex-1));
        addSuggestion(sb.toString());

        sb = new StringBuilder(suggestedWord);
        sb.setCharAt(charIndex+1, suggestedWord.charAt(charIndex));
        sb.setCharAt(charIndex, suggestedWord.charAt(charIndex+1));
        addSuggestion(sb.toString());
    }

    public static void main(String[] args) {

    }

}
