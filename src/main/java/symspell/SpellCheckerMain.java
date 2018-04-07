package symspell;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

public class SpellCheckerMain {
    private int termIndex = 0;
    private int countIndex = 1;
    private String path ="data/dk_freq_dict.txt";
    private SymSpell.Verbosity suggestionVerbosity = SymSpell.Verbosity.All; //Top, Closest, All
    int maxEditDistanceLookup = 3; //max edit distance per lookup (maxEditDistanceLookup<=maxEditDistanceDictionary)
    private SymSpell symSpell;

    public SpellCheckerMain(int maxEditDistanceLookup){
        symSpell = new SymSpell(-1, maxEditDistanceLookup, -1, 10);//, (byte)18);
        this.maxEditDistanceLookup = maxEditDistanceLookup;
        if(!symSpell.loadDictionary(path, termIndex, countIndex)) System.out.println("File not found");
    }

    public List<SuggestItem> lookup(String input){
        return symSpell.lookup(input, suggestionVerbosity, maxEditDistanceLookup);
    }

    public SuggestItem lookupCompound(String input){
        return symSpell.lookupCompound(input, maxEditDistanceLookup).get(0);
    }



    public static void main(String[] args) throws IOException {
        SpellCheckerMain symSpell = new SpellCheckerMain(3);
        //verbosity=Top: the suggestion with the highest term frequency of the suggestions of smallest edit distance found
        //verbosity=Closest: all suggestions of smallest edit distance found, the suggestions are ordered by term frequency
        //verbosity=All: all suggestions <= maxEditDistance, the suggestions are ordered by edit distance, then by term frequency (slower, no early termination)
        // IE All is the only one to give suggestions if a word with exact match is found.

        String inputTerm;
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        while(true){

            System.out.println("Enter input:");
            inputTerm = br.readLine();
//            symspell.SuggestItem suggestions = symSpell.lookupCompound(inputTerm);
            List<SuggestItem> suggestions = symSpell.lookup(inputTerm);
            SuggestItem compound = symSpell.lookupCompound(inputTerm);
//            System.out.println(suggestions.term);
            suggestions.stream()
                    .limit(10)
                    .forEach(suggestion -> System.out.println(suggestion.term + " " + suggestion.distance + " " + suggestion.count));
            System.out.println(compound.term);
        }
    }
}
