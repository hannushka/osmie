package symspell;

import util.StringUtils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class SymSpellTester {
    private static String testFile = "data/manualNameData.csv";

    public static void main(String[] args) throws IOException {
        SymSpell symSpell = new SymSpell(-1, 2,-1, 10);
        if(!symSpell.loadDictionary("data/korpus_freq_dict.txt", 0, 1))
            throw new FileNotFoundException();
        BufferedReader br = new BufferedReader(new FileReader(testFile));
        int editDistOne = 0, noChangeCorrect = 0, noChangeIncorrect = 0, changedCorrectly = 0,
                changedIncorrectly = 0, wrongChangeType = 0;
        String line, correction, in, label;
        String[] split;
        List<SuggestItem> items;
        while((line = br.readLine()) != null){
            split = line.split(",,,");
            in = split[0].toLowerCase().trim();
            label = split[1].toLowerCase().trim();
            items = symSpell.lookupSpecialized(in, SymSpell.Verbosity.All);
            Collections.sort(items);
            if(!items.isEmpty() && items.get(0).distance <= 1)  correction = items.get(0).term;
            else correction = in;
            if(StringUtils.oneEditDist(in, label)) editDistOne++;
            if(in.equals(correction) && in.equals(label)) noChangeCorrect++;
            if(in.equals(correction) && !in.equals(label)) noChangeIncorrect++;
            if(!in.equals(label) && correction.equals(label)) changedCorrectly++;
            if(in.equals(label) && !in.equals(correction)) changedIncorrectly++;
            if(!in.equals(label) && !in.equals(correction) && !correction.equals(label))  wrongChangeType++;
        }
        System.out.println("====");
        System.out.println("Errors (good->bad): \t\t" + changedIncorrectly);
        System.out.println("Corrections (bad->good):\t" + changedCorrectly + " (total #correct: "
                + (changedCorrectly+noChangeCorrect) + ")");
        System.out.println("Correction (bad->bad): \t\t" + wrongChangeType);
        System.out.println("Total changes: \t\t\t\t" + (wrongChangeType + changedCorrectly + changedIncorrectly));
        System.out.println("----");
        System.out.println("Unchanged correctly (good->good): \t" + noChangeCorrect);
        System.out.println("Unchanged incorrectly (bad->bad): \t" + noChangeIncorrect);
        System.out.println("Total unchanged: \t\t\t" + (noChangeCorrect+noChangeIncorrect));
        System.out.println("----");
        System.out.println("Edits within one: \t\t\t" + editDistOne);
    }
}
